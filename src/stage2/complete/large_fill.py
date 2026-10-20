"""Filling holes bigger than one 128-px window (user, 2026-10-20): three strategies, compared in
src/stage2/pipeline/large_fill_compare.py.

    new, info = fill_large(label, hole, "big" | "coarse" | "tiled", fc, fast, generator, t0_refine=150)

label: completion-scale 16-class map (walls 3 px); hole: bool mask (same shape) of the pixels to redraw.
  B1 "big"     the 16-class hybrid (U-Net sketch -> re-noise to t0 -> DDIM, best-of-N) run fully convolutionally on
               ONE square window: the smallest multiple of 32 >= the hole's box + 32 px context on each side (at
               least 128). The networks are diffusers UNet2DModels without position embeddings; self-attention runs
               on the 16 x 16 maps at 128 px and on proportionally larger maps here (global attention, any size).
  B2 "coarse"  Stage 1 first: the plan is mapped 16 -> 6 classes (wall, door, window, room = room types + fixtures,
               outdoor, background; railing -> wall) and framed as in Stage 1 (structure box long side 120 px,
               4 px margin, centred in 128 x 128); the frozen Stage 1 size-adaptive hybrid (configs/app.yaml settings,
               src/interface/inference.Repairer) redraws the hole; its layout is mapped back to completion scale
               (nearest). Inside the hole this layout is the sketch: wall / door / window / outdoor / background
               as drawn, room pixels take the 16-class U-Net's room type / fixture there (else the most common
               room type the U-Net drew in the hole). Then the 16-class hybrid refines it at a LOW t0 (t0_refine,
               150 or 300) on the B1 window, assigning room types / fixtures and sharpening edges.
  B3 "tiled"   overlapping 128-px windows (stride 96, i.e. 32 px overlap) covering the hole's box + 32 px; filled one at
               a time, always next the window with the largest share of known pixels (outside the hole or already
               filled), each window's hole = its still-unfilled hole pixels, so earlier results are context.
t0 of B1 / B3 (own choice, logged): the Stage 1 size-adaptive t0 by the hole's share of the building (400 / 600 /
700; configs/stage2/pipeline.yaml fill.t0 / size_bins), not the small-hole t0_fixed 300, which was chosen on small
holes. Sampling: fast mode (configs/stage2/pipeline.yaml `fast`) when `fast` is given.
After any strategy, region_restrict keeps the full layout (walls, doors, windows, room types, fixtures) except
doors that do not join two interior spaces (no doors on exterior walls).
"""

import time
from functools import lru_cache

import cv2
import numpy as np
import torch

from src.stage2.complete import sampler
from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

I = C16.index
ROOMS = C16.group("S")
FIX = C16.group("F")
CTX = 32
TILE, STRIDE = 128, 96


def _models():
    from src.stage2.pipeline.run import models
    return models()


def crop(arr, oy, ox, side, fill):
    """side x side crop at (oy, ox), padded with `fill` outside the array."""
    out = np.full((side, side), fill, arr.dtype)
    h, w = arr.shape
    y0, x0, y1, x1 = max(0, oy), max(0, ox), min(h, oy + side), min(w, ox + side)
    if y1 > y0 and x1 > x0:
        out[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = arr[y0:y1, x0:x1]
    return out


def paste(dst, src, oy, ox, where):
    """dst[...] = src where `where` (window coords), clipped to dst."""
    ys, xs = np.nonzero(where)
    gy, gx = ys + oy, xs + ox
    ok = (gy >= 0) & (gy < dst.shape[0]) & (gx >= 0) & (gx < dst.shape[1])
    dst[gy[ok], gx[ok]] = src[ys[ok], xs[ok]]


def bbox(m):
    ys, xs = np.nonzero(m)
    return ys.min(), xs.min(), ys.max(), xs.max()


def big_window(hole, ctx=CTX):
    """(oy, ox, side): the smallest multiple-of-32 square (>= 128) holding the hole's box + ctx on each side."""
    y0, x0, y1, x1 = bbox(hole)
    side = max(128, int(np.ceil((max(y1 - y0, x1 - x0) + 1 + 2 * ctx) / 32) * 32))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side


def size_t0(label, hole, fc):
    building = max(1, int(np.isin(label, C16.group("BOUNDARY") + ROOMS + FIX).sum()))
    share = hole.sum() / building
    b = fc["size_bins"]
    return fc["t0"][0] if share < b[0] else (fc["t0"][1] if share < b[1] else fc["t0"][2])


def _cond(win, hole):
    t = torch.from_numpy(win.astype(np.int64))[None]
    target = torch.nn.functional.one_hot(t, C16.n).permute(0, 3, 1, 2).float()
    mask = torch.from_numpy(hole.astype(np.float32))[None, None]
    return torch.cat([target * (1 - mask), mask], 1)


def hybrid_window(label, hole, oy, ox, side, t0, fc, fast, generator, sketch=None):
    """One window: hole pixels redrawn by the 16-class hybrid. sketch: (side, side) labels to use instead of the
    U-Net's. -> (new label map, info, U-Net sketch of the window)."""
    m = _models()
    win = crop(label, oy, ox, side, I["background"])
    wh = crop(hole.astype(np.uint8), oy, ox, side, 0) > 0
    cond = _cond(win, wh)
    with torch.no_grad():
        unet_sk = m["unet"].predict(cond)
    sk = unet_sk if sketch is None else torch.from_numpy(sketch.astype(np.int64))[None]
    kw = {"n": fc["n_samples"]} if fast is None else {"n": fast["n_samples"], "steps_from_t0": fast["steps_from_t0"],
                                                       "guide_last": fast["guide_last"]}
    out, info = sampler.fill(m["diffusion"], cond, sketch=sk, t0=t0, eta_guidance=fc["eta_guidance"],
                             steps=fc["ddim_steps"], generator=generator, **kw)
    new = label.copy()
    paste(new, out[0].numpy().astype(np.uint8), oy, ox, wh)
    return new, info[0], unet_sk[0].numpy().astype(np.uint8)


def fill_big(label, hole, fc, fast, generator):
    oy, ox, side = big_window(hole)
    t0 = size_t0(label, hole, fc)
    new, info, _ = hybrid_window(label, hole, oy, ox, side, t0, fc, fast, generator)
    return new, {"window": side, "t0": t0, **info}


def tiles(hole, ctx=CTX, tile=TILE, stride=STRIDE):
    y0, x0, y1, x1 = bbox(hole)

    def starts(a, b):
        span = (b - a + 1) + 2 * ctx
        n = max(1, int(np.ceil((span - tile) / stride)) + 1)
        lo, hi = a - ctx, b + ctx + 1 - tile
        return sorted({int(round(v)) for v in np.linspace(lo, max(lo, hi), n)})
    return [(oy, ox) for oy in starts(y0, y1) for ox in starts(x0, x1)]


def fill_tiled(label, hole, fc, fast, generator):
    t0 = size_t0(label, hole, fc)
    new, left, order = label.copy(), hole.copy(), []
    cand = tiles(hole)
    while left.any():
        best, best_known = None, -1.0
        for oy, ox in cand:
            part = crop(left.astype(np.uint8), oy, ox, TILE, 0) > 0
            if not part.any():
                continue
            known = 1.0 - part.mean()
            if known > best_known:
                best, best_known = (oy, ox), known
        if best is None:                                  # hole pixels outside every tile (should not happen)
            break
        oy, ox = best
        part = np.zeros(hole.shape, bool)
        paste(part, np.ones((TILE, TILE), bool), oy, ox, crop(left.astype(np.uint8), oy, ox, TILE, 0) > 0)
        new, _, _ = hybrid_window(new, part, oy, ox, TILE, t0, fc, fast, generator)
        left &= ~part
        order.append(best)
    return new, {"tiles": len(order), "t0": t0}


# ---------------------------------------------------------------- B2: Stage 1 coarse layout

S1_ROOM, S1 = 3, {"wall": 0, "door": 1, "window": 2, "room": 3, "outdoor": 4, "background": 5}
TO6 = np.full(C16.n, S1["background"], np.uint8)
for _c in range(C16.n):
    _n = C16.names[_c]
    TO6[_c] = (S1[_n] if _n in S1 else S1["wall"] if _n == "railing" else S1["room"] if _c in ROOMS + FIX
               else S1["outdoor"] if _n == "outdoor" else S1["background"])
FROM6 = np.array([I["wall"], I["door"], I["window"], 255, I["outdoor"], I["background"]], np.uint8)   # 255: room


@lru_cache(maxsize=None)
def stage1_repairer():
    """The frozen Stage 1 size-adaptive hybrid (configs/app.yaml sampling); checkpoints from
    configs/stage2/pipeline.yaml checkpoints.stage1_unet / stage1_diffusion."""
    from src.interface.inference import Repairer, load_config
    from src.stage2.pipeline.run import cfg as pcfg
    acfg = load_config()
    c = pcfg()["checkpoints"]
    acfg["checkpoints"] = {"unet_matched": c["stage1_unet"], "diffusion": c["stage1_diffusion"]}
    return Repairer(acfg, device=torch.device("cpu"))


def frame6(label):
    """Affine (2 x 3) mapping completion px -> Stage 1 frame (structure long side 120 px, centred, 4 px margin)."""
    ys, xs = np.nonzero(np.isin(label, [I["wall"], I["door"], I["window"]]))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = 120.0 / max(y1 - y0, x1 - x0)
    ty = 4 + (120 - (y1 - y0) * s) / 2 - y0 * s
    tx = 4 + (120 - (x1 - x0) * s) / 2 - x0 * s
    return np.array([[s, 0, tx], [0, s, ty]], np.float32)


def stage1_layout(label, hole, seed=0):
    """Stage 1 hybrid on the framed 6-class plan -> (layout at completion scale: 16-class ids for wall / door /
    window / outdoor / background, 255 for room; seconds)."""
    A = frame6(label)
    plan6 = cv2.warpAffine(TO6[label], A, (128, 128), flags=cv2.INTER_NEAREST, borderValue=S1["background"])
    m6 = cv2.warpAffine(hole.astype(np.float32), A, (128, 128), flags=cv2.INTER_LINEAR) > 0.25
    t = time.time()
    rep = stage1_repairer().repair(plan6, m6, "size_adaptive_hybrid", seed=seed)
    back = cv2.warpAffine(rep.labels, A, label.shape[::-1], flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderValue=S1["background"])
    return FROM6[back], time.time() - t, rep.details


def fill_coarse(label, hole, fc, fast, generator, t0_refine=150, seed=0):
    layout, s1_sec, details = stage1_layout(label, hole, seed)
    oy, ox, side = big_window(hole)
    wh = crop(hole.astype(np.uint8), oy, ox, side, 0) > 0
    win = crop(label, oy, ox, side, I["background"])
    with torch.no_grad():
        unet_sk = _models()["unet"].predict(_cond(win, wh))[0].numpy().astype(np.uint8)
    lay = crop(layout, oy, ox, side, I["background"])
    room_px = unet_sk[wh & np.isin(unet_sk, ROOMS)]
    default_room = int(np.bincount(room_px).argmax()) if len(room_px) else I["other_room"]
    room_like = np.where(np.isin(unet_sk, ROOMS + FIX), unet_sk, default_room)
    sk = np.where(wh, np.where(lay == 255, room_like, lay), win).astype(np.uint8)
    new, info, _ = hybrid_window(label, hole, oy, ox, side, t0_refine, fc, fast, generator, sketch=sk)
    return new, {"window": side, "t0_refine": t0_refine, "stage1_s": round(s1_sec, 1), "stage1": details, **info}


def region_restrict(before, after, hole):
    """Full layout kept inside the hole, except doors that do not join two interior spaces (reverted)."""
    from src.stage2.pipeline.run import interior_door
    out = np.where(hole, after, before)
    for d in components((out != before) & (out == I["door"])):
        if not interior_door(out, d):
            out[d] = before[d]
    return out


def fill_large(label, hole, strategy, fc, fast=None, generator=None, t0_refine=150):
    t = time.time()
    if strategy == "big":
        new, info = fill_big(label, hole, fc, fast, generator)
    elif strategy == "coarse":
        new, info = fill_coarse(label, hole, fc, fast, generator, t0_refine)
    elif strategy == "tiled":
        new, info = fill_tiled(label, hole, fc, fast, generator)
    else:
        raise ValueError(strategy)
    info["seconds"] = round(time.time() - t, 1)
    return region_restrict(label, new, hole), info

"""Novelty step 2 - training samples cut from the large-omission set (src/novelty/build_set.py).

    samples = load_set(["large_train_0.zip", "large_train_1.zip"])
    FillSet(samples, win=256)        -> (cond, target, mask) windows for the large-hole FILL models
    LocateSet(samples, canvas=320)   -> (one-hot parsed map, missing-area mask) for the region LOCATOR

A sample is one version of one drawing: the parser's map of it (`map`), the ground truth at the same size (`gt`)
and the omission map (`om`: 0 nothing, 1 omission area, 2 erased structure). Version 0 is the intact drawing.

Fill windows (what the large-hole models learn from)
  * erased version: hole = the bounding box of the omission area, each side moved by -2..+8 px (the box a user
    draws, or the locator predicts, is never exact);
  * intact version: a random large area chosen on the ground truth exactly as B4 did (a whole room 2-15% or a
    rectangle 5-25% of the building, src/stage2/complete/windows.large_deletion), same box rule.
  context (outside the hole) = the PARSER's map; target = ground truth inside the hole, the parser's map outside.
  So the model learns to draw the true layout into the noisy map it will really be given.
Locator samples: the whole parsed map on a fixed canvas (random position, cropped if larger); target = the
omission area (all zeros for an intact drawing).
"""

import csv
import io
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.stage2.complete.windows import LARGE_TYPES, large_deletion, model_inputs
from src.stage2.data.classes16 import C16

BG = C16.index["background"]


def dec(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


def load_set(paths, limit=None):
    """[{pid, version, kind, size_pct, size_bin, map, gt, om}] with the three images kept as PNG bytes (small)."""
    out = []
    for path in paths:
        with zipfile.ZipFile(path) as z:
            for r in csv.DictReader(io.StringIO(z.read("meta.csv").decode("utf-8"))):
                stem = f"{r['pid']}_v{r['version']}"
                out.append({"pid": r["pid"], "version": int(r["version"]), "kind": r["kind"],
                            "size_pct": float(r["size_pct"] or 0), "size_bin": r["size_bin"],
                            "map": z.read(f"{stem}_map.png"), "gt": z.read(f"{stem}_gt.png"),
                            "om": z.read(f"{stem}_om.png")})
                if limit and len(out) >= limit:
                    return out
    return out


def arrays(s):
    return dec(s["map"]), dec(s["gt"]), dec(s["om"])


def bbox(m):
    ys, xs = np.nonzero(m)
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def crop(a, oy, ox, h, w, fill):
    """h x w crop at (oy, ox), padded with `fill` outside the array."""
    out = np.full((h, w), fill, a.dtype)
    H, W = a.shape
    y0, x0, y1, x1 = max(0, oy), max(0, ox), min(H, oy + h), min(W, ox + w)
    if y1 > y0 and x1 > x0:
        out[y0 - oy:y1 - oy, x0 - ox:x1 - ox] = a[y0:y1, x0:x1]
    return out


def augment(arrs, rng):
    k, flip = int(rng.integers(4)), rng.random() < 0.5
    return [np.ascontiguousarray(np.rot90(a[:, ::-1] if flip else a, k)) for a in arrs]


# ---------------------------------------------------------------- fill windows

JITTER = (-2, 9)          # each side of the hole box moves by -2..+8 px (rng.integers upper bound is exclusive)


def hole_box(area, shape, rng=None):
    """Bounding box of the omission area, jittered per side when rng is given, clipped to the map."""
    y0, x0, y1, x1 = bbox(area)
    if rng is not None:
        j = rng.integers(*JITTER, size=4)
        y0, x0, y1, x1 = y0 - j[0], x0 - j[1], y1 + j[2], x1 + j[3]
        if y1 - y0 < 8 or x1 - x0 < 8:                     # jitter must not collapse a small area
            y0, x0, y1, x1 = bbox(area)
    return max(0, y0), max(0, x0), min(shape[0] - 1, y1), min(shape[1] - 1, x1)


def fill_window(parsed, gt, area, rng, win=256, jitter=True, centre=False):
    """(context window, target window, hole mask) or None when the hole does not fit the window.
    centre=True puts the hole in the middle (validation / inference); otherwise at a random position."""
    y0, x0, y1, x1 = hole_box(area, parsed.shape, rng if jitter else None)
    if y1 - y0 + 1 > win - 16 or x1 - x0 + 1 > win - 16:
        return None
    if centre:
        oy, ox = (y0 + y1) // 2 - win // 2, (x0 + x1) // 2 - win // 2
    else:
        oy, ox = int(rng.integers(y1 - win + 9, y0 - 7)), int(rng.integers(x1 - win + 9, x0 - 7))
    hole = np.zeros(parsed.shape, np.uint8)
    hole[y0:y1 + 1, x0:x1 + 1] = 1
    ctx, tgt, hw = crop(parsed, oy, ox, win, win, BG), crop(gt, oy, ox, win, win, BG), crop(hole, oy, ox, win, win, 0)
    return ctx, np.where(hw > 0, tgt, ctx), hw


def sample_fill(s, rng, win=256):
    """One training window from a sample (see the module docstring), or None."""
    parsed, gt, om = arrays(s)
    if s["version"] == 0:
        for kind in rng.permutation(LARGE_TYPES):
            area = large_deletion(gt, str(kind), rng)
            if area is not None and area.any():
                break
        else:
            return None
    else:
        area = om > 0
    w = fill_window(parsed, gt, area, rng, win)
    return None if w is None else augment(w, rng)


def to_model(ctx, tgt, hole):
    """Windows -> cond (17, H, W) = one-hot context with the hole zeroed + mask, target (16, H, W), mask (1, H, W)."""
    _, target, mask = model_inputs(tgt[None], hole[None])
    known = torch.nn.functional.one_hot(torch.from_numpy(ctx.astype(np.int64)), C16.n).permute(2, 0, 1).float()
    cond = torch.cat([known * (1 - mask[0]), mask[0]], 0)
    return cond, target[0], mask[0]


class FillSet(Dataset):
    """Endless random fill windows; intact_frac of them come from intact versions (random areas)."""

    def __init__(self, samples, win=256, intact_frac=0.4, length=10 ** 7):
        self.erased = [s for s in samples if s["version"] > 0]
        self.intact = [s for s in samples if s["version"] == 0]
        self.win, self.intact_frac, self.length, self.rng = win, intact_frac, length, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        while True:
            pool = self.intact if (self.intact and self.rng.random() < self.intact_frac) else self.erased
            w = sample_fill(pool[self.rng.integers(len(pool))], self.rng, self.win)
            if w is not None:
                return to_model(*w)


def fixed_fill_windows(samples, n, seed=1234, win=256):
    """n reproducible validation windows from ERASED versions (hole = the exact area box, centred):
    (cond, target, mask) tensors and the list of sample indices used."""
    rng = np.random.default_rng(seed)
    erased = [i for i, s in enumerate(samples) if s["version"] > 0]
    out, used = [], []
    for i in rng.permutation(erased):
        parsed, gt, om = arrays(samples[i])
        w = fill_window(parsed, gt, om > 0, rng, win, jitter=False, centre=True)
        if w is not None:
            out.append(to_model(*w))
            used.append(int(i))
        if len(out) >= n:
            break
    cond, target, mask = (torch.stack(t) for t in zip(*out))
    return cond, target, mask, used


# ---------------------------------------------------------------- locator samples

def locate_canvas(parsed, area, rng, canvas=320):
    """Parsed map + area mask on a canvas x canvas square: random offset when it fits, else a random crop
    (through the area when there is one)."""
    H, W = parsed.shape

    def origin(n, lo, hi):
        if n <= canvas:
            return -int(rng.integers(0, canvas - n + 1))
        if lo is None:
            return int(rng.integers(0, n - canvas + 1))
        c = int(rng.integers(lo, hi + 1))
        return int(np.clip(c - int(rng.integers(canvas // 4, 3 * canvas // 4)), 0, n - canvas))
    if area.any():
        y0, x0, y1, x1 = bbox(area)
    else:
        y0 = x0 = y1 = x1 = None
    oy, ox = origin(H, y0, y1), origin(W, x0, x1)
    return crop(parsed, oy, ox, canvas, canvas, BG), crop(area.astype(np.uint8), oy, ox, canvas, canvas, 0)


class LocateSet(Dataset):
    """Endless locator samples; erased_frac of them have an omission."""

    def __init__(self, samples, canvas=320, erased_frac=0.6, length=10 ** 7):
        self.erased = [s for s in samples if s["version"] > 0]
        self.intact = [s for s in samples if s["version"] == 0]
        self.canvas, self.erased_frac, self.length, self.rng = canvas, erased_frac, length, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        pool = self.erased if (not self.intact or self.rng.random() < self.erased_frac) else self.intact
        parsed, _, om = arrays(pool[self.rng.integers(len(pool))])
        lab, tgt = augment(locate_canvas(parsed, om > 0, self.rng, self.canvas), self.rng)
        return torch.from_numpy(lab.astype(np.int64)), torch.from_numpy(tgt.astype(np.float32))[None]

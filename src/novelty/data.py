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


def load_set(paths, limit=None, max_version=None):
    """[{pid, version, kind, size_pct, size_bin, map, gt, om, ink}] with the images kept as PNG bytes (small).
    ink (where the drawing has any ink, build_set.ink_map) is None for sets built without --ink.
    max_version: keep only versions 0..max_version of every drawing (3 = exactly the first, smaller set)."""
    out = []
    for path in paths:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            for r in csv.DictReader(io.StringIO(z.read("meta.csv").decode("utf-8"))):
                if max_version is not None and int(r["version"]) > max_version:
                    continue
                stem = f"{r['pid']}_v{r['version']}"
                out.append({"pid": r["pid"], "version": int(r["version"]), "kind": r["kind"],
                            "size_pct": float(r["size_pct"] or 0), "size_bin": r["size_bin"],
                            "map": z.read(f"{stem}_map.png"), "gt": z.read(f"{stem}_gt.png"),
                            "om": z.read(f"{stem}_om.png"),
                            "ink": z.read(f"{stem}_ink.png") if f"{stem}_ink.png" in names else None})
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


def fill_window(parsed, gt, area, rng, win=256, jitter=True, centre=False, box=None):
    """(context window, target window, hole mask) or None when the hole does not fit the window.
    centre=True puts the hole in the middle (validation / inference); otherwise at a random position.
    box: use this hole box (e.g. the locator's) instead of the box of `area`."""
    y0, x0, y1, x1 = box if box is not None else hole_box(area, parsed.shape, rng if jitter else None)
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


def sample_fill(s, rng, win=256, loc_frac=0.0):
    """One training window from a sample (see the module docstring), or None.
    loc_frac: share of erased-version windows whose hole is the LOCATOR's box for that omission (s["loc_box"],
    set by train_fill.add_locator_boxes) instead of the jittered true box - what the pipeline really hands over."""
    parsed, gt, om = arrays(s)
    box = s.get("loc_box") if (s["version"] > 0 and loc_frac and rng.random() < loc_frac) else None
    if s["version"] == 0:
        for kind in rng.permutation(LARGE_TYPES):
            area = large_deletion(gt, str(kind), rng)
            if area is not None and area.any():
                break
        else:
            return None
    else:
        area = om > 0
    w = fill_window(parsed, gt, area, rng, win, box=box)
    return None if w is None else augment(w, rng)


def to_model(ctx, tgt, hole, hide=True):
    """Windows -> cond (17, H, W) = one-hot context + mask, target (16, H, W), mask (1, H, W).
    hide=True  ("inpaint"): the context is zeroed inside the hole, as in Stage 1 / B4;
    hide=False ("edit"): the parser's map stays visible inside the hole - the model corrects it instead of
                redrawing it blind (the merged room still shows hints of the missing one, and an area where
                nothing is missing can simply be kept)."""
    _, target, mask = model_inputs(tgt[None], hole[None])
    known = torch.nn.functional.one_hot(torch.from_numpy(ctx.astype(np.int64)), C16.n).permute(2, 0, 1).float()
    cond = torch.cat([known * (1 - mask[0]) if hide else known, mask[0]], 0)
    return cond, target[0], mask[0]


class FillSet(Dataset):
    """Endless random fill windows; intact_frac of them come from intact versions (random areas)."""

    def __init__(self, samples, win=256, intact_frac=0.4, length=10 ** 7, hide=True, loc_frac=0.0):
        self.erased = [s for s in samples if s["version"] > 0]
        self.intact = [s for s in samples if s["version"] == 0]
        self.win, self.intact_frac, self.length, self.rng = win, intact_frac, length, None
        self.hide, self.loc_frac = hide, loc_frac

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        while True:
            pool = self.intact if (self.intact and self.rng.random() < self.intact_frac) else self.erased
            w = sample_fill(pool[self.rng.integers(len(pool))], self.rng, self.win, self.loc_frac)
            if w is not None:
                return to_model(*w, hide=self.hide)


def fixed_fill_windows(samples, n, seed=1234, win=256, hide=True):
    """n reproducible validation windows from ERASED versions (hole = the exact area box, centred):
    (cond, target, mask) tensors and the list of sample indices used."""
    rng = np.random.default_rng(seed)
    erased = [i for i, s in enumerate(samples) if s["version"] > 0]
    out, used = [], []
    for i in rng.permutation(erased):
        parsed, gt, om = arrays(samples[i])
        w = fill_window(parsed, gt, om > 0, rng, win, jitter=False, centre=True)
        if w is not None:
            out.append(to_model(*w, hide=hide))
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


# ---------------------------------------------------------------- structure-net samples (round 2)

def ray_bytes(parsed):
    """Wall-continuation rays of a parsed map as PNG bytes: 0 none, 1 junction ray, 2 free ray (src/novelty/rays.py)."""
    from src.novelty.rays import continuation_rays
    free, junction = continuation_rays(parsed)
    return cv2.imencode(".png", (junction.astype(np.uint8) + 2 * free.astype(np.uint8)))[1].tobytes()


class StructureSet(Dataset):
    """Endless samples for the structure net (src/novelty/structure.py): the whole parsed map on a canvas
    (as LocateSet) -> (labels int64 (H, W), extra float (3, H, W) = free rays, junction rays, ink,
    targets float (2, H, W) = missing-wall band, missing area). erased_frac of the samples have an omission.
    Needs s["rays"] (structure.add_rays); s["ink"] may be None (zeros)."""

    def __init__(self, samples, canvas=320, erased_frac=0.7, length=10 ** 7):
        self.erased = [s for s in samples if s["version"] > 0]
        self.intact = [s for s in samples if s["version"] == 0]
        self.canvas, self.erased_frac, self.length, self.rng = canvas, erased_frac, length, None

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        from src.novelty.structure import targets
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
        pool = self.erased if (not self.intact or self.rng.random() < self.erased_frac) else self.intact
        s = pool[self.rng.integers(len(pool))]
        parsed, gt, om = arrays(s)
        rays = dec(s["rays"])
        ink = dec(s["ink"]) if s.get("ink") is not None else np.zeros(parsed.shape, np.uint8)
        band, area = targets(gt, om)
        H, W = parsed.shape
        c = self.canvas

        def origin(n, lo, hi):                           # as locate_canvas: random offset, or a crop through the area
            if n <= c:
                return -int(self.rng.integers(0, c - n + 1))
            if lo is None:
                return int(self.rng.integers(0, n - c + 1))
            m = int(self.rng.integers(lo, hi + 1))
            return int(np.clip(m - int(self.rng.integers(c // 4, 3 * c // 4)), 0, n - c))
        if area.any():
            y0, x0, y1, x1 = bbox(area > 0)
        else:
            y0 = x0 = y1 = x1 = None
        oy, ox = origin(H, y0, y1), origin(W, x0, x1)
        lab, r, k, b, a = augment([crop(parsed, oy, ox, c, c, BG), crop(rays, oy, ox, c, c, 0), crop(ink, oy, ox, c, c, 0),
                                   crop(band.astype(np.uint8), oy, ox, c, c, 0), crop(area.astype(np.uint8), oy, ox, c, c, 0)],
                                  self.rng)
        extra = np.stack([(r == 2), (r == 1), k.astype(np.float32) / 255]).astype(np.float32)
        return (torch.from_numpy(lab.astype(np.int64)), torch.from_numpy(extra),
                torch.from_numpy(np.stack([b, a]).astype(np.float32)))

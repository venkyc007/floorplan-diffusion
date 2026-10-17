"""16-class training windows for the FILL models: a 128 x 128 crop of a completion-set
plan (walls 3 px) around one random typed deletion.

Deletion types (as in Stage 1, plus railing and fixture):
  door, window, railing, fixture   one 4-connected component of that class
  wall                             one straight wall piece 8-40 px long (a horizontal or
                                   vertical run of the wall map, cut at a random point)
  room                             one room component (+ its boundary within 3 px), if it fits
  region                           a random rectangle, sides 8-64 px
Mask = the deleted pixels' bounding box + random 2-6 px padding per side (Stage 1 rule:
never the exact shape, which would leak the answer), clipped to the window.

Returns label window (H, W) uint8, mask (H, W) uint8 (1 = hole), deletion type.
model_inputs() turns them into cond (17 ch: one-hot with the hole zeroed + mask) and target (16 ch).
"""

import cv2
import numpy as np
import torch

from src.stage2.data.classes16 import C16

TYPES = ["door", "window", "wall", "railing", "fixture", "room", "region"]
I = C16.index
FIX = C16.group("F")
ROOMS = C16.group("S")


def _components(mask):
    n, comp, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    return [(comp == i, st[i]) for i in range(1, n)]


def _bbox(m):
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _wall_piece(label, rng):
    """A straight wall piece: run of a 1x9 (or 9x1) opening, cut to 8-40 px."""
    wall = (label == I["wall"]).astype(np.uint8)
    horizontal = rng.random() < 0.5
    k = np.ones((1, 9) if horizontal else (9, 1), np.uint8)
    runs = cv2.morphologyEx(wall, cv2.MORPH_OPEN, k)
    comps = _components(runs)
    if not comps:
        return None
    m, _ = comps[rng.integers(len(comps))]
    y0, y1, x0, x1 = _bbox(m)
    length = int(rng.integers(8, 41))
    piece = np.zeros_like(m)
    if horizontal:
        s = int(rng.integers(x0, max(x0 + 1, x1 - length + 2)))
        piece[y0:y1 + 1, s:s + length] = m[y0:y1 + 1, s:s + length]
    else:
        s = int(rng.integers(y0, max(y0 + 1, y1 - length + 2)))
        piece[s:s + length, x0:x1 + 1] = m[s:s + length, x0:x1 + 1]
    return piece if piece.any() else None


def deletion(label, kind, rng, win=128):
    """Bool mask of the pixels removed by one deletion of `kind`, or None if impossible."""
    if kind in ("door", "window", "railing", "fixture"):
        cls = FIX if kind == "fixture" else [I[kind]]
        comps = _components(np.isin(label, cls))
        return comps[rng.integers(len(comps))][0] if comps else None
    if kind == "wall":
        return _wall_piece(label, rng)
    if kind == "room":
        comps = [m for m, st in _components(np.isin(label, ROOMS))
                 if st[cv2.CC_STAT_WIDTH] <= win - 20 and st[cv2.CC_STAT_HEIGHT] <= win - 20 and st[cv2.CC_STAT_AREA] >= 20]
        if not comps:
            return None
        m = comps[rng.integers(len(comps))]
        ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        return m | (ring & np.isin(label, C16.group("BOUNDARY")))
    if kind == "region":
        inside = np.argwhere(~np.isin(label, C16.group("BG")))
        if len(inside) == 0:
            return None
        cy, cx = inside[rng.integers(len(inside))]
        h, w = rng.integers(8, 65, size=2)
        m = np.zeros(label.shape, bool)
        m[max(0, cy - h // 2):cy - h // 2 + h, max(0, cx - w // 2):cx - w // 2 + w] = True
        return m
    raise KeyError(kind)


def sample_window(label, rng, win=128, types=TYPES, augment=True):
    """(window labels, mask, type) or None. The hole lies fully inside the window when it fits."""
    order = list(rng.permutation(types))
    for kind in order:
        removed = deletion(label, kind, rng, win)
        if removed is None or not removed.any():
            continue
        y0, y1, x0, x1 = _bbox(removed)
        if kind != "region":                      # Stage 1 rule: bbox + 2-6 px padding per side
            pad = rng.integers(2, 7, size=4)
            y0, y1, x0, x1 = y0 - pad[0], y1 + pad[1], x0 - pad[2], x1 + pad[3]
        h, w = label.shape
        y0, x0, y1, x1 = max(0, y0), max(0, x0), min(h - 1, y1), min(w - 1, x1)
        if y1 - y0 + 1 > win or x1 - x0 + 1 > win:
            continue
        # window origin so the hole is inside it, at a random position
        oy = int(rng.integers(y1 - win + 1, y0 + 1))
        ox = int(rng.integers(x1 - win + 1, x0 + 1))
        lab = np.full((win, win), I["background"], np.uint8)
        sy0, sx0, sy1, sx1 = max(0, oy), max(0, ox), min(h, oy + win), min(w, ox + win)
        lab[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox] = label[sy0:sy1, sx0:sx1]
        mask = np.zeros((win, win), np.uint8)
        mask[y0 - oy:y1 - oy + 1, x0 - ox:x1 - ox + 1] = 1
        if augment:
            k = int(rng.integers(4))
            lab, mask = np.rot90(lab, k), np.rot90(mask, k)
            if rng.random() < 0.5:
                lab, mask = lab[:, ::-1], mask[:, ::-1]
        return np.ascontiguousarray(lab), np.ascontiguousarray(mask), kind
    return None


def model_inputs(labels, masks):
    """Batch of window labels (B, H, W) + masks (B, H, W) -> cond (B, 17, H, W), target (B, 16, H, W), mask (B, 1, H, W)."""
    lab = torch.as_tensor(np.asarray(labels)).long()
    target = torch.nn.functional.one_hot(lab, C16.n).permute(0, 3, 1, 2).float()
    mask = torch.as_tensor(np.asarray(masks)).float()[:, None]
    cond = torch.cat([target * (1 - mask), mask], 1)
    return cond, target, mask

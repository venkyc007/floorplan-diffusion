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


# ---------------------------------------------------------------- large holes (B4, user 2026-10-21)

LARGE_TYPES = ["room_whole", "region_large"]
LARGE_ROOMS = ["bedroom", "bathroom", "kitchen", "other_room"]
WALL3 = 3                                         # completion-scale wall thickness


def _building(label):
    return ~np.isin(label, C16.group("OUT"))


def large_deletion(label, kind, rng, room_pct=(2.0, 15.0), region_pct=(5.0, 25.0)):
    """Label-level twin of src/stage2/data/erasure_large.py at completion scale (walls 3 px):
    room_whole    a bedroom / bathroom / kitchen / other_room instance of room_pct % of the building + its interior
                  walls (within 2 wall thicknesses, not within 2 of outdoor / background), the doors in them, its
                  windows and fixtures;
    region_large  a rectangle (aspect 0.6-1.6) covering region_pct % of the building.
    -> bool mask of the removed pixels (region: the rectangle) or None."""
    b = _building(label)
    nb = max(int(b.sum()), 1)
    if kind == "room_whole":
        from src.stage2.constraints.rules import room_instances
        rooms = [m for c, m in room_instances(label) if C16.names[c] in LARGE_ROOMS
                 and room_pct[0] <= 100.0 * m.sum() / nb <= room_pct[1]]
        if not rooms:
            return None
        room = rooms[rng.integers(len(rooms))]
        grow = lambda m, r: cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0
        near, outer = grow(room, 2 * WALL3), grow(np.isin(label, C16.group("OUT")), 2 * WALL3)
        out = room | (near & (label == I["wall"]) & ~outer)
        for m, _ in _components(label == I["door"]):
            if (m & grow(room, 3)).any():
                out |= m
        for m, _ in _components(label == I["window"]):
            if (m & near).sum() >= 0.5 * m.sum():
                out |= m
        for m, _ in _components(np.isin(label, FIX)):
            if (m & grow(room, 2)).sum() >= 0.5 * m.sum():
                out |= m
        return out
    if kind == "region_large":
        ys, xs = np.nonzero(b)
        if len(ys) == 0:
            return None
        for _ in range(20):
            target = rng.uniform(*region_pct) / 100 * nb
            aspect = rng.uniform(0.6, 1.6)
            h, w = int(np.sqrt(target * aspect)), int(np.sqrt(target / aspect))
            k = rng.integers(len(ys))
            m = np.zeros(label.shape, bool)
            m[max(0, ys[k] - h // 2):ys[k] - h // 2 + h, max(0, xs[k] - w // 2):xs[k] - w // 2 + w] = True
            if region_pct[0] <= 100.0 * (m & b).sum() / nb <= region_pct[1]:
                return m
        return None
    raise KeyError(kind)


def sample_large_window(label, rng, win=256, augment=True):
    """(window labels, mask, type) with ONE large hole (the removed pixels' box + 2-6 px, as Stage 1) inside the
    window, or None."""
    for kind in rng.permutation(LARGE_TYPES):
        removed = large_deletion(label, kind, rng)
        if removed is None or not removed.any():
            continue
        y0, y1, x0, x1 = _bbox(removed)
        pad = rng.integers(2, 7, size=4)
        h, w = label.shape
        y0, x0, y1, x1 = max(0, y0 - pad[0]), max(0, x0 - pad[2]), min(h - 1, y1 + pad[1]), min(w - 1, x1 + pad[3])
        if y1 - y0 + 1 > win - 16 or x1 - x0 + 1 > win - 16:
            continue
        oy = int(rng.integers(y1 - win + 9, y0 - 7))
        ox = int(rng.integers(x1 - win + 9, x0 - 7))
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
        return np.ascontiguousarray(lab), np.ascontiguousarray(mask), str(kind)
    return None


def sample_mixed_window(label, rng, win=256, large_frac=0.5):
    """B4 training mix: large_frac large holes, the rest the original small component holes (same window size)."""
    if rng.random() < large_frac:
        w = sample_large_window(label, rng, win)
        if w is not None:
            return w
    return sample_window(label, rng, win)

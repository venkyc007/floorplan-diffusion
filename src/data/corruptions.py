"""Removing parts of a floor plan in a controlled way ("corruptions").

The model's task (mask-given inpainting): it gets a plan where a region is
blanked out, plus a mask saying WHERE that region is, and must redraw what
was inside. Everything outside the mask is known.

Corruption types:
  door    one door component
  window  one window component
  wall    one straight wall segment between two junctions/corners
  room    one room-interior component with its surrounding walls/doors/windows
  region  a random rectangle on the building

The mask must not give the answer away, so it is NEVER the exact shape of
what was removed: it is the removed part's bounding box, grown by a random
2-6 px on each side (each side independently). For "region" the rectangle
itself is the mask.

Main entry points:
  corrupt(label, ctype, rng, cfg)        -> (mask, corrupted, meta) or None
  corrupt_multi(label, types, rng, cfg)  -> same, several parts at once
  to_model_input(corrupted, mask)        -> 7-channel model input
"""

import cv2
import numpy as np

from src.data.classes import (BACKGROUND, DOOR, NUM_CLASSES, OUTDOOR, ROOM,
                              WALL, WINDOW)

# In a corrupted label map, masked pixels get this value ("unknown").
UNKNOWN = 255
TYPES = ["door", "window", "wall", "room", "region"]
BUILDING = [WALL, DOOR, WINDOW, ROOM]   # pixels that make up the building
BOUNDARY = [WALL, DOOR, WINDOW]         # walls + openings = room boundaries
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)  # 4-neighbourhood


# ------------------------------------------------------------- helpers

def components(mask):
    """Find the connected pieces of a True/False mask (4-connectivity).

    Returns (piece_image, stats): piece_image numbers every pixel by its
    piece (0 = not in the mask); stats[i] = (x, y, width, height, area) of
    piece i. Pieces are only turned into full masks once one is chosen,
    which keeps this fast.
    """
    _, pieces, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    return pieces, stats


def bbox(mask):
    """(row0, col0, row1, col1) of the True pixels; row1/col1 are exclusive."""
    rows, cols = np.nonzero(mask)
    return rows.min(), cols.min(), rows.max() + 1, cols.max() + 1


def padded_box_mask(part, rng, padding, shape):
    """Mask = bounding box of `part`, each side pushed out by its own random
    padding (padding[0]..padding[1] px), clipped to the image."""
    r0, c0, r1, c1 = bbox(part)
    lo, hi = padding
    top, left, bottom, right = rng.integers(lo, hi + 1, size=4)
    mask = np.zeros(shape, bool)
    mask[max(0, r0 - top):min(shape[0], r1 + bottom),
         max(0, c0 - left):min(shape[1], c1 + right)] = True
    return mask


def touches(part, label, classes):
    """True if any pixel next to `part` (4-adjacency) has one of `classes`."""
    grown = cv2.dilate(part.astype(np.uint8), CROSS).astype(bool)
    return bool(np.isin(label[grown & ~part], classes).any())


# ------------------------------------------- finding removable components

# Each *_candidates function returns a list of (piece_image, piece_number)
# pairs: one entry per part that may be removed.

def opening_candidates(label, cls):
    """Every door (or window) component."""
    pieces, stats = components(label == cls)
    return [(pieces, i) for i in range(1, len(stats))]


def wall_segments(label, run_length, length_range):
    """Straight wall pieces between junctions/corners.

    Walls are axis-aligned and ~2 px thick. A wall pixel is "horizontal" if
    it lies on a horizontal run of >= run_length wall pixels (found with a
    morphological opening), "vertical" likewise. Pixels that are both are
    corners/junctions; removing them splits the walls into straight pieces.
    Openings (doors/windows) also end a piece, since they are not wall.
    Keeps pieces whose length (longer bounding-box side) is in length_range.
    """
    wall = (label == WALL).astype(np.uint8)
    horiz = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((1, run_length), np.uint8)).astype(bool)
    vert = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((run_length, 1), np.uint8)).astype(bool)
    lo, hi = length_range
    out = []
    for straight in (horiz & ~vert, vert & ~horiz):
        pieces, stats = components(straight)
        out += [(pieces, i) for i in range(1, len(stats))
                if lo <= max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) <= hi]
    return out


def room_candidates(label, min_area, max_box_pct):
    """Room-interior components (see with_room_boundary for what is removed).

    Rooms joined by open passages form one big component; if its bounding
    box covers more than max_box_pct % of the building it is "most of the
    plan", not one room, and is skipped.
    """
    max_box = max_box_pct / 100 * np.isin(label, BUILDING).sum()
    pieces, stats = components(label == ROOM)
    return [(pieces, i) for i in range(1, len(stats))
            if stats[i, cv2.CC_STAT_AREA] >= min_area
            and stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT] <= max_box]


def with_room_boundary(label, room):
    """The room plus the wall/door/window pixels within 3 px of it (walls
    are ~2 px thick), so a whole room "cell" disappears."""
    ring = cv2.dilate(room.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return room | (ring & np.isin(label, BOUNDARY))


def region_mask(label, rng, side_range):
    """A random rectangle centred on a random building pixel."""
    rows, cols = np.nonzero(np.isin(label, BUILDING))
    k = rng.integers(len(rows))
    h, w = rng.integers(side_range[0], side_range[1] + 1, size=2)
    r0, c0 = rows[k] - h // 2, cols[k] - w // 2
    mask = np.zeros(label.shape, bool)
    mask[max(0, r0):max(0, r0 + h), max(0, c0):max(0, c0 + w)] = True
    return mask


def pick_part(label, ctype, rng, cfg):
    """Choose what to remove. Returns (removed_pixels, mask) or None if the
    plan has nothing valid of this type."""
    if ctype == "region":
        mask = region_mask(label, rng, cfg["region"]["side"])
        return mask & np.isin(label, BUILDING), mask
    if ctype == "door":
        cands = opening_candidates(label, DOOR)
    elif ctype == "window":
        cands = opening_candidates(label, WINDOW)
    elif ctype == "wall":
        cands = wall_segments(label, cfg["wall"]["run_length"], cfg["wall"]["segment_length"])
    elif ctype == "room":
        cands = room_candidates(label, cfg["room"]["min_area"], cfg["room"]["max_box_pct"])
    else:
        raise ValueError(f"unknown corruption type {ctype!r}")
    if not cands:
        return None
    pieces, i = cands[rng.integers(len(cands))]
    part = pieces == i
    if ctype == "room":
        part = with_room_boundary(label, part)
    return part, padded_box_mask(part, rng, cfg["mask"]["padding"], label.shape)


# ------------------------------------------------------------ metadata

def describe(label, mask, removed, cfg):
    """Metadata of one corruption: location, position, size bin, areas."""
    building = np.isin(label, BUILDING)
    # Exterior = the removed part touches outdoor or background.
    exterior = removed.any() and touches(removed, label, [OUTDOOR, BACKGROUND])

    # Position: mask centre's distance from the building centre, divided by
    # the building "radius" (largest centre-to-building-pixel distance).
    br, bc = np.nonzero(building)
    cr, cc = br.mean(), bc.mean()
    radius = np.sqrt((br - cr) ** 2 + (bc - cc) ** 2).max()
    mr, mc = np.nonzero(mask)
    dist = float(np.hypot(mr.mean() - cr, mc.mean() - cc) / radius)

    area_pct = 100.0 * mask.sum() / building.sum()
    small, large = cfg["size_bins"]["edges"]
    return {
        "location": "exterior" if exterior else "interior",
        "position": "edge" if dist >= cfg["position"]["edge_cutoff"] else "center",
        "size_bin": "small" if area_pct < small else ("medium" if area_pct < large else "large"),
        "center_dist": round(dist, 4),
        "mask_area": int(mask.sum()),
        "mask_pct": round(float(area_pct), 3),
    }


# ---------------------------------------------------------- main API

def apply_mask(label, mask):
    """Corrupted label map: masked pixels become UNKNOWN."""
    corrupted = label.copy()
    corrupted[mask] = UNKNOWN
    return corrupted


def corrupt(label, ctype, rng, cfg):
    """Remove one part of type `ctype` from a clean label map.

    Returns (mask, corrupted, meta), or None if the plan has no valid part
    of that type (the caller counts these skips).
    """
    picked = pick_part(label, ctype, rng, cfg)
    if picked is None:
        return None
    removed, mask = picked
    meta = {"type": ctype, "n_components": 1, **describe(label, mask, removed, cfg)}
    return mask, apply_mask(label, mask), meta


def corrupt_multi(label, types, rng, cfg):
    """Remove several parts at once (one per entry of `types`); the mask is
    the union of their masks. Types with no valid part are left out; returns
    None if nothing could be removed."""
    mask = np.zeros(label.shape, bool)
    removed = np.zeros(label.shape, bool)
    done = []
    for ctype in types:
        picked = pick_part(label, ctype, rng, cfg)
        if picked is not None:
            removed |= picked[0]
            mask |= picked[1]
            done.append(ctype)
    if not done:
        return None
    meta = {"type": "multi", "subtypes": "+".join(done), "n_components": len(done),
            **describe(label, mask, removed, cfg)}
    return mask, apply_mask(label, mask), meta


def to_model_input(corrupted, mask):
    """Model input (7, H, W) float32: the 6 one-hot channels with the masked
    area set to all zeros, plus the mask itself as channel 7."""
    onehot = (np.arange(NUM_CLASSES)[:, None, None] == corrupted[None]).astype(np.float32)
    return np.concatenate([onehot, mask[None].astype(np.float32)])

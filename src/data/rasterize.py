"""Turning one vector floor plan into a 128x128 image ("rasterizing").

Steps for one plan:
  1. Find the bounding box of the building (walls, doors, rooms, balconies).
  2. Scale it so the LONGER side fills the image minus the padding, keeping
     the aspect ratio, and centre it.
  3. Paint each class in PAINT_ORDER into a "label map": a 128x128 array where
     every pixel holds one class number (0=wall, 1=door, ... 5=background).
  4. Convert the label map to 6 channels ("one-hot"): channel c is 1 where
     the pixel is class c, else 0.
"""

import cv2
import numpy as np

from src.data.classes import (BACKGROUND, EXTENT_FIELDS, NUM_CLASSES,
                              PAINT_ORDER, RESPLAN_FIELDS)
from src.data.resplan import get_polygons

# cv2 draws with integer coordinates. Multiplying by 2**SHIFT first keeps
# sub-pixel precision (important: walls are only ~2 pixels thick).
SHIFT = 4


def plan_bounds(plan):
    """Smallest box (xmin, ymin, xmax, ymax) containing the building."""
    boxes = [poly.bounds
             for field in EXTENT_FIELDS
             for poly in get_polygons(plan, field)]
    boxes = np.array(boxes)
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


def make_transform(bounds, size, padding):
    """Return (scale, function) that maps plan coordinates to pixel coordinates.

    The plan's longer side becomes (size - 2*padding) pixels; the plan is
    centred. The y axis is flipped because in images row 0 is at the TOP,
    while in the plan data y grows UPWARDS.
    """
    xmin, ymin, xmax, ymax = bounds
    scale = (size - 2 * padding) / max(xmax - xmin, ymax - ymin)
    # Leftover space on each side, so the plan sits in the middle.
    off_x = (size - (xmax - xmin) * scale) / 2
    off_y = (size - (ymax - ymin) * scale) / 2

    def to_pixels(coords):
        coords = np.asarray(coords)
        cols = (coords[:, 0] - xmin) * scale + off_x
        rows = (ymax - coords[:, 1]) * scale + off_y   # flip y
        # -0.5: cv2 treats integer coordinates as pixel CENTRES.
        pts = np.stack([cols, rows], axis=1) - 0.5
        return np.round(pts * (1 << SHIFT)).astype(np.int32)

    return scale, to_pixels


def draw_polygons(polygons, size, to_pixels):
    """Draw polygons into a True/False mask, leaving their holes empty."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [to_pixels(poly.exterior.coords)], 1, shift=SHIFT)
        for hole in poly.interiors:
            cv2.fillPoly(mask, [to_pixels(hole.coords)], 0, shift=SHIFT)
    return mask.astype(bool)


def rasterize_plan(plan, size=128, padding=4):
    """Convert one ResPlan plan into a label map (size x size, values 0..5).

    Also returns the scale (pixels per plan unit), saved in the metadata.
    """
    scale, to_pixels = make_transform(plan_bounds(plan), size, padding)
    label = np.full((size, size), BACKGROUND, dtype=np.uint8)
    for cls in PAINT_ORDER:
        polygons = [p for field in RESPLAN_FIELDS[cls]
                    for p in get_polygons(plan, field)]
        label[draw_polygons(polygons, size, to_pixels)] = cls
    return label, scale


def to_onehot(label):
    """Label map (H, W) -> one-hot array (6, H, W) of 0s and 1s (uint8)."""
    return (np.arange(NUM_CLASSES)[:, None, None] == label[None]).astype(np.uint8)


def to_label(onehot):
    """One-hot array (6, H, W) -> label map (H, W). Inverse of to_onehot."""
    return onehot.argmax(axis=0).astype(np.uint8)

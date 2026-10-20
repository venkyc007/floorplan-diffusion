"""Gap at a flag (user, 2026-10-20): how long is the break in the wall line where a door flag points?

    g = flagged_gap(label, box, point)    # -> {"length", "mask", "horizontal"} or None

Used by the pipeline to tell "door missing" from "wall missing" for door flags (R5, door-ink, detector):
  length > Lmax_door or < Lmin_door  -> no door fits: the wall line is closed with wall ("wall missing")
  Lmin_door <= length <= Lmax_door   -> door logic as before
Completion scale (walls 3 px). For each direction the wall mask is opened with a 5 px line (real runs only, not walls crossing it) and
closed with a line of 2 * Lmax_door + 1 px, so breaks up to twice the longest door between two ALIGNED runs are
bridged. Gap = bridged pixels that are not wall or window (a parsed door counts as part of the gap: the opening
it sits in). Gap components that span 2-6 px across the wall and touch the flag box (grown 1 px) and lie within 3 px of the
flag point are kept; the nearest one wins. Length = the longest run on one row (column) of the wall, not the box (a
diagonal wall would give a long box of short runs). None: no break in a wall line at the flag (e.g. the doorway was drawn closed).
"""

import cv2
import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.data.classes16 import C16

I = C16.index


RUN = 5              # a wall run must be >= 5 px long in its direction (a 3-4 px thick wall across it is not a run)
NEAR = 3             # the gap must be within 3 px of the flag point (the flag points at it)
MAX_ACROSS = 6       # a break THROUGH a wall line is at most ~2 wall thicknesses wide (bigger = a room bridged)


def flagged_gap(label, box, point=None, const=CONST):
    hi = const["Lmax_door"]
    y0, x0, y1, x1 = box
    py, px = point if point is not None else ((y0 + y1) // 2, (x0 + x1) // 2)
    pad = 2 * hi + 4                                       # work on a crop around the flag
    H, W = label.shape
    cy0, cx0, cy1, cx1 = max(0, y0 - pad), max(0, x0 - pad), min(H, y1 + pad + 1), min(W, x1 + pad + 1)
    lab = label[cy0:cy1, cx0:cx1]
    wall = (lab == I["wall"]).astype(np.uint8)
    solid = np.isin(lab, [I["wall"], I["window"]])
    near_box = np.zeros(lab.shape, bool)
    near_box[max(0, y0 - cy0 - 1):y1 - cy0 + 2, max(0, x0 - cx0 - 1):x1 - cx0 + 2] = True
    best = None
    for horiz in (True, False):
        runs = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((1, RUN) if horiz else (RUN, 1), np.uint8))
        k = np.ones((1, 2 * hi + 1) if horiz else (2 * hi + 1, 1), np.uint8)
        gap = (cv2.morphologyEx(runs, cv2.MORPH_CLOSE, k) > 0) & ~solid
        n, comp, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=4)
        for i in range(1, n):
            m = comp == i
            per_line = m.sum(1) if horiz else m.sum(0)           # gap length on each row (column) of the wall
            along, across = int(per_line.max()), int((per_line > 0).sum())
            if not 2 <= across <= MAX_ACROSS or not (m & near_box).any():
                continue
            ys, xs = np.nonzero(m)
            d = float(np.min((ys + cy0 - py) ** 2 + (xs + cx0 - px) ** 2))
            if d <= NEAR ** 2 and (best is None or d < best[0]):
                full = np.zeros(label.shape, bool)
                full[cy0:cy1, cx0:cx1] = m
                best = (d, {"length": int(along), "mask": full, "horizontal": horiz})
    return best[1] if best else None


def gap_verdict(length, const=CONST):
    """'wall' if no door fits the gap (shorter than Lmin_door or longer than Lmax_door), else 'door'."""
    return "door" if const["Lmin_door"] <= length <= const["Lmax_door"] else "wall"

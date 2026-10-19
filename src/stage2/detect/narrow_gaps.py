"""R7 "gap too narrow for a doorway" (user, 2026-10-19): a break in a straight wall run with wall on both ends,
aligned, SHORTER than the shortest door (Lmin_door), and not labelled door / window -> "wall missing".

    cands = gap_candidates_narrow(label, min_width=1, min_run=5)

Completion scale (walls 3 px). For each direction, the wall mask is opened with a straight line of `min_run` px
(only real wall runs, not corner stubs), then closed with a line of Lmin_door px: the closing fills breaks shorter
than Lmin_door between two ALIGNED runs. Gap = filled pixels that are not wall, door or window in the parse.
A gap component counts if it spans at least 2 px across the wall (a break through the wall, not a notch on one
face) and is at least `min_width` px long. Box = the gap. Sketch prior: wall closing the gap (sketch_prior.close_wall).
"""

import cv2
import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.data.classes16 import C16
from src.stage2.detect.rules import Candidate, box_of

I = C16.index


def gap_candidates_narrow(label, min_width=1, min_run=5, const=CONST):
    wall = (label == I["wall"]).astype(np.uint8)
    opening = np.isin(label, [I["wall"], I["door"], I["window"]])
    L = const["Lmin_door"]
    out = []
    for horiz in (True, False):
        runs = cv2.morphologyEx(wall, cv2.MORPH_OPEN, np.ones((1, min_run) if horiz else (min_run, 1), np.uint8))
        closed = cv2.morphologyEx(runs, cv2.MORPH_CLOSE, np.ones((1, L) if horiz else (L, 1), np.uint8)) > 0
        gap = closed & ~opening
        n, comp, stats, _ = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=4)
        for i in range(1, n):
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            along, across = (w, h) if horiz else (h, w)
            if across >= 2 and min_width <= along < L:
                m = comp == i
                out.append(Candidate("wall", box_of(m, label.shape), ["R7"], float(along)))
    return out

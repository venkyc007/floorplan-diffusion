"""Doorless-gap rule (R5, user): a doorway left open in a wall where real plans would have a door.

    cands = gap_candidates(label)      # -> [rules.Candidate(type "door", rules ["R5"])]

A gap = a place where two DIFFERENT spaces touch directly (4-neighbours) across a short straight line:
  - the contact line is horizontal or vertical, its length is in [Lmin_door, Lmax_door] (data constants),
  - there is wall at both ends of the line (within 3 px along it), i.e. it is a break in a wall run,
  - the pair of spaces is one that real plans almost never leave open: direct-contact rate < 10% in
    outputs/stage2/data_stats.md (c6_contact_rates). Open-plan pairs (living-hallway, living-kitchen,
    kitchen-hallway, living-other_room, kitchen-other_room) and outdoor-background are >= 10% and left alone.
Bedroom / bathroom openings without a door occur in only ~1-3% of real plan adjacencies, so such a gap
is far more likely a forgotten door than a design choice.
"""

import cv2
import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.data.classes16 import C16
from src.stage2.detect.rules import Candidate, box_of

I = C16.index
SPACES = C16.group("S") + C16.group("OUT")
OPEN_RATE = 0.10
END_SEARCH = 3


def closed_pairs(const=CONST, open_rate=OPEN_RATE, skip=()):
    """(16, 16) bool: True for space pairs whose direct contact rate is below open_rate
    (pairs with a class in `skip` are never flagged)."""
    rates = const.get("c6_contact_rates", {})
    bad = np.zeros((C16.n, C16.n), bool)
    for a in SPACES:
        for b in SPACES:
            na, nb = C16.names[a], C16.names[b]
            if a == b or na in skip or nb in skip:
                continue
            r = rates.get(f"{na}-{nb}", rates.get(f"{nb}-{na}", 0.0))
            bad[a, b] = r < open_rate
    return bad


def _runs(line):
    """Connected runs (8-connectivity) of a contact-line image -> list of masks."""
    n, comp = cv2.connectedComponents(line.astype(np.uint8), connectivity=8)
    return [comp == i for i in range(1, n)]


def gap_candidates(label, const=CONST, open_rate=OPEN_RATE, skip=()):
    lo, hi = const["Lmin_door"], const["Lmax_door"]
    bad = closed_pairs(const, open_rate, skip)
    wall = label == I["wall"]
    h, w = label.shape
    out = []
    # horizontal lines: contact between (y, x) and (y + 1, x); vertical: between (y, x) and (y, x + 1)
    for axis in (0, 1):
        a = label[:-1, :] if axis == 0 else label[:, :-1]
        b = label[1:, :] if axis == 0 else label[:, 1:]
        line = np.zeros((h, w), bool)
        hit = bad[a, b]
        if axis == 0:
            line[:-1, :] = hit
        else:
            line[:, :-1] = hit
        for m in _runs(line):
            ys, xs = np.nonzero(m)
            along = xs if axis == 0 else ys
            across = ys if axis == 0 else xs
            length = int(along.max() - along.min() + 1)
            if not lo <= length <= hi or across.max() - across.min() > 3:      # short and straight
                continue
            c0, c1 = int(across.min()) - 1, int(across.max()) + 3                  # band across the line
            s0, s1 = int(along.min()), int(along.max())
            if axis == 0:
                end0 = wall[max(0, c0):c1, max(0, s0 - END_SEARCH):s0]
                end1 = wall[max(0, c0):c1, s1 + 1:s1 + 1 + END_SEARCH]
            else:
                end0 = wall[max(0, s0 - END_SEARCH):s0, max(0, c0):c1]
                end1 = wall[s1 + 1:s1 + 1 + END_SEARCH, max(0, c0):c1]
            if end0.any() and end1.any():
                out.append(Candidate("door", box_of(m, label.shape), ["R5"], float(length)))
    return out

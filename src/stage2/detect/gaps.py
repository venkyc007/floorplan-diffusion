"""Doorless-gap rule (R5, user): a doorway left open in a wall where real plans would have a door.

    cands = gap_candidates(label)      # -> [rules.Candidate(type "door", rules ["R5"])]

A gap = a place where two DIFFERENT spaces touch directly (4-neighbours) across a short straight line:
  - the contact line is horizontal or vertical, its length is in [Lmin_door, Lmax_door] (data constants),
  - there is wall at both ends of the line (within 3 px along it), i.e. it is a break in a wall run,
  - the pair of spaces is one that real plans almost never leave open: direct-contact rate < 10% in
    outputs/stage2/data_stats.md (c6_contact_rates). Open-plan pairs (living-hallway, living-kitchen,
    kitchen-hallway, living-other_room, kitchen-other_room) and outdoor-background are >= 10% and left alone.
  - final version (user, overnight): one side is a bedroom, a bathroom, or an other_room at least as
    large as the 25th-percentile bedroom (generic "Room" / "UserDefined" spaces are often bedrooms), and
    the other side is not a small other_room (< 25th-percentile other_room area: closets). Cut-offs in
    configs/stage2/detector_threshold.yaml (gap_rule), chosen on the tuning plans (detect/gap_tune.py).
Bedroom / bathroom openings without a door occur in only ~1-3% of real plan adjacencies, so such a gap
is far more likely a forgotten door than a design choice. Known limitation: on a drawing, the rule can
only be as good as the parser's room types.
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


def instance_areas(label):
    """(H, W) area of the room instance (4-connected component of one room type) at each pixel, 0 elsewhere."""
    area = np.zeros(label.shape, np.int64)
    for c in C16.group("S"):
        n, comp, stats, _ = cv2.connectedComponentsWithStats((label == c).astype(np.uint8), connectivity=4)
        if n > 1:
            a = stats[:, cv2.CC_STAT_AREA].copy()
            a[0] = 0
            area += a[comp]
    return area


def needs_door(label, area, ys, xs, ys2, xs2, need_area, small_other):
    """Final R5 condition (user): one side is a bedroom, a bathroom, or an other_room of at least
    `need_area` px (generic rooms are often bedrooms); the other side is not a small other_room
    (< `small_other` px: closets). Sides = most common class on each face of the contact line."""
    sides = []
    for yy, xx in ((ys, xs), (ys2, xs2)):
        c = int(np.bincount(label[yy, xx]).argmax())
        sides.append((C16.names[c], int(np.median(area[yy, xx]))))

    def big(s):
        return s[0] in ("bedroom", "bathroom") or (s[0] == "other_room" and s[1] >= need_area)

    def closet(s):
        return s[0] == "other_room" and s[1] < small_other

    a, b = sides
    return (big(a) and not closet(b)) or (big(b) and not closet(a))


def gap_candidates(label, const=CONST, open_rate=OPEN_RATE, skip=(), need_area=None, small_other=None,
                   min_len=None):
    """need_area / small_other (px): the final R5 room-size condition (needs_door); None = pair rule only.
    min_len: shortest gap (px) that fires (default Lmin_door)."""
    lo, hi = const["Lmin_door"], const["Lmax_door"]
    lo = max(lo, min_len or 0)
    bad = closed_pairs(const, open_rate, skip)
    area = instance_areas(label) if need_area is not None else None
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
            if not (end0.any() and end1.any()):
                continue
            if area is not None:                 # faces: the line pixel and its neighbour across it
                ys2, xs2 = (ys + 1, xs) if axis == 0 else (ys, xs + 1)
                if not needs_door(label, area, ys, xs, ys2, xs2, need_area, small_other):
                    continue
            out.append(Candidate("door", box_of(m, label.shape), ["R5"], float(length)))
    return out

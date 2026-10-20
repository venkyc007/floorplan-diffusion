"""R8 "oversized / merged room" -> "region missing" (user, 2026-10-20; v2 relative to the drawing 2026-10-21).

    cands = merged_room_candidates(label, k_type=1.8, k_all=2.5, mix=True)   # [Candidate("region", ...)]
    rooms = merged_room_scores(label)                                        # per room, for tuning

v2 (user): a room instance (4-connected region of one room type, completion scale, >= MIN_PX) is flagged when
  size   its area > k_type x the median area of the OTHER rooms of the same type in this drawing, or, when the type
         appears only once, > k_all x the median area of all the drawing's rooms (the room's own area excluded
         from both medians, so a merged room cannot raise its own reference); or
  mix    it holds fixtures that no single room type takes: the rooms allowed for each fixture (C15 table:
         fixture_rooms) have an empty intersection, e.g. a toilet / bath (bathroom) AND an appliance (kitchen).
         "other_room" is left out of the C15 sets here (it accepts every fixture, so no intersection would ever
         be empty). A fixture counts for a room if >= half of it lies in the room grown 2 px.
v1 (p99 of the type, R1 table) did not fire on held-out merges (0 / 20): merged rooms stay under p99.
The flag's box = the room's bounding box + MARGIN px; the pipeline sends that whole box to the large-hole fill.
Score = 2 + area ratio, so a region flag wins any merge with overlapping small flags (scores <= 1).
"""

import cv2
import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.detect import rules as R

I = C16.index
FIXTURES = list(C16.cfg["groups"]["F"])
ALLOWED = {f: set(r) - {"other_room"} for f, r in CONST.get("fixture_rooms", {}).items()}
MIN_PX = 50
MARGIN = 4
OPEN = ("living", "hallway")     # open-plan spaces: often several times the median room in real plans


def _grown(m, r):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def merged_room_scores(label):
    """[(mask, ratio, single, mixed, type name)]: ratio = area / reference median (same type, or all rooms if
    single)."""
    rooms = [(c, m, int(m.sum())) for c, m in room_instances(label) if m.sum() >= MIN_PX]
    fx = [(C16.names[int(np.bincount(label[m]).argmax())], m)
          for m in components(np.isin(label, [I[f] for f in FIXTURES]))]
    out = []
    for i, (c, m, a) in enumerate(rooms):
        same = [b for j, (c2, _, b) in enumerate(rooms) if c2 == c and j != i]
        others = [b for j, (_, _, b) in enumerate(rooms) if j != i]
        single = not same
        ref = float(np.median(same)) if same else (float(np.median(others)) if others else np.inf)
        near = _grown(m, 2)
        kinds = {f for f, fm in fx if (fm & near).sum() >= 0.5 * fm.sum()}
        sets = [ALLOWED[f] for f in kinds if f in ALLOWED]
        mixed = bool(sets) and not set.intersection(*sets)
        out.append((m, a / ref, single, mixed, C16.names[c]))
    return out


def flagged(ratio, single, mixed, k_type, k_all, mix, name=None, skip_open=False):
    """skip_open: the size test ignores living rooms and hallways (the fixture test still applies)."""
    size = ratio > (k_all if single else k_type) and not (skip_open and name in OPEN)
    return size or (mix and mixed)


def merged_room_candidates(label, k_type=1.8, k_all=2.5, mix=True, rooms=None, skip_open=False):
    out = []
    for m, ratio, single, mixed, name in merged_room_scores(label) if rooms is None else rooms:
        if flagged(ratio, single, mixed, k_type, k_all, mix, name, skip_open):
            ys, xs = np.nonzero(m)
            out.append(R.Candidate("region", R.box_of(m, label.shape, pad=MARGIN), ["R8"], 2.0 + min(ratio, 10.0),
                                   (int(np.median(ys)), int(np.median(xs))), m))
    return out

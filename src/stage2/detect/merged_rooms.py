"""R8 "oversized / merged room" (user, 2026-10-20): a parsed room that is really two or more rooms whose dividing
walls are missing -> "region missing", box = the room's bounding box.

    cands = merged_room_candidates(label, area_mult=1.0, mix=True)   # [Candidate("region", box, ["R8"], score)]
    rooms = merged_room_scores(label)                                # [(mask, area / p99, mixed?)] for tuning

A room instance (4-connected region of one room type, completion scale) is flagged when
  size   its area > area_mult x the p99 area of that room type (R1 table: constraint_constants room_area), or
  mix    it holds fixtures that no single room type takes: the rooms allowed for each fixture (C15 table:
         fixture_rooms) have an empty intersection, e.g. a toilet / bath (bathroom) AND an appliance (kitchen).
         Own choice: "other_room" is left out of the C15 sets here; it accepts every fixture, so with it no
         intersection is ever empty. A fixture counts for a room if >= half of it lies in the room grown 2 px.
Score = area / p99 (>= 1 for size flags; mix-only flags get 1.0), so merging keeps the biggest.
"""

import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.constraints.rules import components, room_instances
from src.stage2.data.classes16 import C16
from src.stage2.detect import rules as R

I = C16.index
FIXTURES = list(C16.cfg["groups"]["F"])
P99 = {k: v[1] for k, v in CONST.get("room_area", {}).items()}
ALLOWED = {f: set(r) - {"other_room"} for f, r in CONST.get("fixture_rooms", {}).items()}


def _grown(m, r):
    import cv2
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)) > 0


def merged_room_scores(label):
    fx = [(C16.names[int(np.bincount(label[m]).argmax())], m) for m in components(np.isin(label, [I[f] for f in FIXTURES]))]
    out = []
    for c, m in room_instances(label):
        name = C16.names[c]
        ratio = float(m.sum() / P99[name]) if name in P99 else 0.0
        near = _grown(m, 2)
        kinds = {f for f, fm in fx if (fm & near).sum() >= 0.5 * fm.sum()}
        sets = [ALLOWED[f] for f in kinds if f in ALLOWED]
        mixed = bool(sets) and not set.intersection(*sets)
        out.append((m, ratio, mixed))
    return out


def merged_room_candidates(label, area_mult=1.0, mix=True, rooms=None):
    out = []
    for m, ratio, mixed in merged_room_scores(label) if rooms is None else rooms:
        if ratio > area_mult or (mix and mixed):
            ys, xs = np.nonzero(m)
            out.append(R.Candidate("region", R.box_of(m, label.shape, pad=1), ["R8"], max(ratio, 1.0),
                                   (int(np.median(ys)), int(np.median(xs))), m))
    return out

"""Check-only rules R1-R4 (spec section 1): verifier and detection, never losses.

They need whole rooms or counts, so they work on discrete label maps (H, W) of class
indices (numpy), not on probabilities. Each returns (violated: bool, mask: (H, W) bool)
where the mask marks the offending pixels (empty when nothing is wrong).

Constants they use (`room_area`, `forbidden_door_pairs`) come from
configs/stage2/constraint_constants.yaml; until the CubiCasa statistics provide them,
the defaults below are placeholders (R1 then checks nothing).
"""

import cv2
import numpy as np

from src.stage2.constraints.constraints import CONST
from src.stage2.data.classes16 import C16
from src.stage2.data.plan_stats import door_side_pair, fill_fixtures

ROOMS = C16.group("S")
DEFAULT_FORBIDDEN = [["bathroom", "kitchen"]]     # only used if the constants file has no list


def components(mask):
    n, comp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    return [comp == i for i in range(1, n)]


def room_instances(label):
    """[(class index, bool mask)] for every room instance: a 4-connected region of one room
    type INCLUDING the fixtures standing in it (fixtures take the nearest room type first)."""
    filled = fill_fixtures(label)
    return [(c, m) for c in ROOMS for m in components(filled == c)]


def r1_room_area(label, const=CONST):
    """R1: room instance area outside its type's [min, max] range (px at completion scale)."""
    ranges = const.get("room_area", {})
    bad = np.zeros(label.shape, bool)
    for c, m in room_instances(label):
        r = ranges.get(C16.names[c])
        if r and not (r[0] <= m.sum() <= r[1]):
            bad |= m
    return bool(bad.any()), bad


def room_count(label, region=None):
    """Number of room instances not touching `region` (all of them if region is None)."""
    return sum(1 for _, m in room_instances(label) if region is None or not (m & region).any())


def r2_room_count(pred, ref, hole, grow=3):
    """R2: the rooms away from the hole must stay the same: equal number of room instances
    that do not touch the hole (grown by `grow` px) in the prediction and the reference."""
    region = cv2.dilate(hole.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)) > 0
    diff = room_count(pred, region) - room_count(ref, region)
    return diff != 0, np.zeros(pred.shape, bool) if diff == 0 else ~region


def _door_sides(label, door, k):
    """Space classes found k px beyond the door on each face (both axes)."""
    ys, xs = np.nonzero(door)
    h, w = label.shape
    found = set()
    for dy, dx in ((k, 0), (-k, 0), (0, k), (0, -k)):
        y, x = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
        found.update(np.unique(label[y, x]).tolist())
    return found


def r3_adjacency(label, const=CONST):
    """R3: a door joining a forbidden pair of space types (e.g. bathroom entered from the kitchen).

    The pair is read exactly as in the statistics (plan_stats.door_side_pair), so the
    forbidden list (pairs on < 0.5% of real door connections) applies one to one.
    """
    k = const["T_wall"] + 2
    pairs = {tuple(sorted((C16.index[a], C16.index[b])))
             for a, b in const.get("forbidden_door_pairs", DEFAULT_FORBIDDEN)}
    bad = np.zeros(label.shape, bool)
    for door in components(label == C16.index["door"]):
        if door_side_pair(label, door, k) in pairs:
            bad |= door
    return bool(bad.any()), bad


def r4_exterior_door(label, const=CONST):
    """R4: the plan needs at least one door with a room on one face and the outside on the other."""
    k = const["T_wall"] + 2
    outside = set(C16.group("OUT"))
    for door in components(label == C16.index["door"]):
        sides = _door_sides(label, door, k)
        if sides & outside and sides & set(ROOMS):
            return False, np.zeros(label.shape, bool)
    return True, np.isin(label, C16.group("W"))      # no single location: mark the walls


RULES = {"R1": r1_room_area, "R3": r3_adjacency, "R4": r4_exterior_door}   # R2 also needs ref + hole

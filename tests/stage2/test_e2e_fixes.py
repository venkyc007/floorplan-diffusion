"""Fixes from the end-to-end figure (user, 2026-10-20): wall-gap rule, symbol already in the parse, door at the
flag point, one room type per room hole, door kept with its adjacent new walls."""

import numpy as np

from src.stage2.complete.sketch_prior import place_opening
from src.stage2.data.classes16 import C16
from src.stage2.detect.flag_gap import flagged_gap, gap_verdict
from src.stage2.detect.rules import Candidate
from src.stage2.pipeline.run import existing, flag_point, gap_rule, restrict

I = C16.index


def two_rooms(gap_len):
    """Bedroom above, hallway below, a horizontal wall (rows 30-32) with a break of gap_len px from x = 40."""
    lab = np.full((64, 128), I["bedroom"], np.uint8)
    lab[33:] = I["hallway"]
    lab[30:33] = I["wall"]
    lab[30:33, 40:40 + gap_len] = I["hallway"]
    return lab


def test_gap_length_and_verdict():
    for n, verdict in ((12, "door"), (40, "wall"), (3, "wall")):
        g = flagged_gap(two_rooms(n), (30, 40, 32, 40 + n - 1))
        assert g is not None and g["horizontal"] and g["length"] == n
        assert gap_verdict(g["length"]) == verdict


def test_no_gap_when_wall_is_closed():
    lab = two_rooms(0)
    assert flagged_gap(lab, (30, 40, 32, 52)) is None


def test_long_gap_door_flag_becomes_wall():
    lab = two_rooms(40)
    h = {"type": "door", "rules": ["R5"], "box": (30, 40, 32, 79), "point": (31, 60)}
    gap_rule(lab, h)
    assert h["type"] == "wall" and h["gap_len"] == 40 and h["gap"][31, 60]
    h = {"type": "door", "rules": ["INK"], "box": (30, 40, 32, 51), "point": (31, 45)}
    gap_rule(two_rooms(12), h)
    assert h["type"] == "door" and h["gap_len"] == 12
    h = {"type": "door", "rules": ["C8"], "box": (30, 40, 32, 79), "point": None}
    gap_rule(lab, h)
    assert h["type"] == "door"                                    # C8 flags have no location: untouched


def test_flag_point():
    assert flag_point(Candidate("door", (0, 0, 10, 20), ["C8"])) is None
    assert flag_point(Candidate("door", (0, 0, 10, 20), ["INK"])) == (5, 10)
    assert flag_point(Candidate("door", (0, 0, 10, 20), ["detector"], 0.9, (2, 3))) == (2, 3)


def test_existing_symbol_in_parse():
    lab = two_rooms(12)
    lab[30:33, 40:52] = I["door"]
    h = {"type": "door", "rules": ["INK"], "box": (30, 40, 32, 51), "point": (31, 45)}
    m = existing(lab, h)
    assert m is not None and m.sum() == 36
    assert existing(lab, dict(h, point=None)) is None             # no location: never marked
    assert existing(two_rooms(12), h) is None                      # no door there


def test_door_goes_on_the_wall_at_the_point():
    """Two wall lines in the hole: without a point the longer one gets the door, with a point the one at it."""
    sk = np.full((64, 64), I["bedroom"], np.uint8)
    sk[10:13, 0:64] = I["wall"]                                   # long horizontal wall
    sk[13:30, 40:43] = I["wall"]                                  # shorter vertical wall below it
    hole = np.zeros((64, 64), bool)
    hole[5:60, 20:50] = True
    no_point = place_opening(sk, hole, I["door"])
    assert (no_point[10:13] == I["door"]).any()
    at_point = place_opening(sk, hole, I["door"], point=(25, 41))
    door = at_point == I["door"]
    ys, xs = np.nonzero(door)
    assert door.any() and xs.min() >= 40 and xs.max() <= 42 and abs(int(np.median(ys)) - 25) <= 7


def test_room_restriction_one_room_type_one_door():
    before = np.full((40, 40), I["living"], np.uint8)
    after = before.copy()
    after[20, 5:35] = I["wall"]
    after[21:30, 5:20] = I["bedroom"]
    after[21:25, 20:35] = I["kitchen"]                            # fewer pixels than bedroom
    after[20, 8:12] = I["door"]
    after[20, 25:29] = I["door"]
    kept = restrict(before, after, "room", point=(20, 26))
    assert (kept == I["bedroom"]).sum() == (after == I["bedroom"]).sum()
    assert not (kept == I["kitchen"]).any()
    doors = kept == I["door"]
    assert doors[20, 25:29].all() and not doors[20, 8:12].any()    # one door, the one at the point
    assert (kept[20, 13:24] == I["wall"]).all()


def test_door_keeps_adjacent_new_wall():
    before = np.full((40, 40), I["bedroom"], np.uint8)
    before[20:23] = I["wall"]
    after = before.copy()
    after[20:23] = I["bedroom"]                                   # the hybrid moved the wall line down 2 px
    after[22:25] = I["wall"]
    after[22:25, 15:28] = I["door"]
    kept = restrict(before, after, "door", point=(23, 21))
    assert (kept[22:25, 15:28] == I["door"]).all()
    assert (kept[22:25, 13:15] == I["wall"]).all()                 # wall next to the door kept
    assert (kept[22:25, 0:10] == I["bedroom"]).all() or (kept[22:25, 0:10] == before[22:25, 0:10]).all()

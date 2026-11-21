"""The structural checks (src/novelty/soundness.py) and the repair (src/novelty/repair.py) on toy maps with one
broken rule each."""

import numpy as np

from src.novelty.repair import repair_fill
from src.novelty.soundness import soundness
from src.stage2.data.classes16 import C16

I = C16.index
WALL, DOOR, WINDOW, LIV, BED, BG = (I[n] for n in ("wall", "door", "window", "living", "bedroom", "background"))


def house():
    """A 60 x 80 living room inside a 10 px background ring, 3 px walls, a front door in the bottom wall.
    Region / hole = the room's inside."""
    m = np.full((80, 100), BG, np.uint8)
    m[10:70, 10:90] = WALL
    m[13:67, 13:87] = LIV
    m[67:70, 45:55] = DOOR
    region = np.zeros(m.shape, bool)
    region[13:67, 13:87] = True
    return m, region


def test_intact_house_is_sound():
    m, region = house()
    r = soundness(m, region)
    assert r["sound"], r


def test_wall_stub_is_a_dead_end():
    m, region = house()
    m[13:40, 40:43] = WALL                                # from the top wall, stops in the middle of the room
    r = soundness(m, region)
    assert r["dead_ends"] == 1 and not r["walls_connect"]


def test_floating_piece():
    m, region = house()
    m[35:38, 30:50] = WALL
    r = soundness(m, region)
    assert r["floating"] == 1 and not r["walls_connect"]


def test_door_in_the_middle_of_a_room():
    m, region = house()
    m[40:43, 30:40] = DOOR
    r = soundness(m, region)
    assert r["bad_openings"] == 1 and not r["openings_in_walls"]


def test_room_open_to_the_outside_leaks():
    m, region = house()
    m[30:50, 84:87] = BG                                  # the right wall missing over 20 px ...
    m[30:50, 87:90] = BG
    region[30:50, 84:90] = True
    r = soundness(m, region)
    assert r["leaks"] >= 1 and not r["rooms_closed"]


def test_room_without_a_door():
    m, region = house()
    m[13:67, 50:53] = WALL                                # splits the room; the right half has no door
    m[13:67, 53:87] = BED
    m[67:70, 45:55] = WALL
    m[67:70, 30:40] = DOOR                                # the front door now opens into the left half
    r = soundness(m, region)
    assert r["doorless"] == 1 and not r["rooms_closed"]


def test_repair_gives_the_closed_room_a_door():
    before, region = house()
    after = before.copy()
    after[13:67, 50:53] = WALL
    after[13:67, 53:87] = BED
    after[67:70, 45:55] = WALL
    after[67:70, 30:40] = DOOR
    out, info = repair_fill(before, after, region)
    r = soundness(out, region)
    assert r["doorless"] == 0 and r["sound"], (info, r)
    assert ((out == DOOR) & (after == WALL))[13:67, 50:53].sum() >= 3 * 8   # a door of door length in the new wall


def test_repair_joins_a_floating_piece_and_fills_inside_out_bits():
    before, region = house()
    after = before.copy()
    after[35:38, 30:50] = WALL                            # touches nothing: joined to the nearest wall (the left one)
    after[20:26, 60:70] = BG                              # "outside" left in the middle of the room
    out, _ = repair_fill(before, after, region)
    assert np.isin(out[35:38, 30:50], [WALL, DOOR]).all()      # kept (a door may be cut into it later)
    assert (out[35:38, 13:30] == WALL).all()                    # joined straight to the left wall
    assert (out[20:26, 60:70] != BG).all()
    assert soundness(out, region)["sound"]


def test_repair_removes_a_piece_that_cannot_be_joined():
    before, region = house()
    after = before.copy()
    after[35:38, 30:50] = WALL
    out, _ = repair_fill(before, after, region, {"extend": 10})
    assert (out[35:38, 30:50] == LIV).all()


def test_repair_extends_a_dead_end_to_the_next_wall():
    before, region = house()
    after = before.copy()
    after[13:55, 40:43] = WALL                            # stops 12 px above the bottom wall
    out, _ = repair_fill(before, after, region)
    assert (out[55:67, 40:43] == WALL).all()
    assert soundness(out, region)["dead_ends"] == 0


def test_repair_never_changes_a_sound_fill():
    before, region = house()
    after = before.copy()
    after[13:67, 50:53] = WALL
    after[30:40, 50:53] = DOOR
    out, info = repair_fill(before, after, region)
    assert (out == after).all() and info["fixes"] == []


def test_violation_map_counts_what_the_rule_check_counts():
    """The figures' violation map (assessment_figures.rule_masks) has the same blobs as the project's rule check."""
    from scipy import ndimage

    from src.novelty.assessment_figures import rule_masks
    from src.stage2.pipeline import e2e_val
    m, region = house()
    m[13:40, 40:43] = WALL                                # a dead end and a door in the middle of the room
    m[45:48, 60:70] = DOOR
    box = (13, 13, 66, 86)
    masks = rule_masks(m, box)
    counts = {k: ndimage.label(v, np.ones((3, 3)))[1] for k, v in masks.items()}
    expected = e2e_val.rule_violations(m, box)
    assert {k: v for k, v in counts.items() if v} == {k: v for k, v in expected.items() if k != "R1"}

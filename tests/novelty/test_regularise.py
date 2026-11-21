"""The clean-up of a region fill (src/novelty/regularise.py): toy maps with one broken rule each."""

import numpy as np

from src.novelty.regularise import checks, regularise
from src.stage2.data.classes16 import C16

I = C16.index
WALL, LIV, BED, KIT, SINK = I["wall"], I["living"], I["bedroom"], I["kitchen"], I["sink"]


def room(h=80, w=100):
    """One living room, 3 px walls all round; the hole = its whole inside."""
    m = np.full((h, w), LIV, np.uint8)
    m[:3], m[-3:], m[:, :3], m[:, -3:] = WALL, WALL, WALL, WALL
    hole = np.zeros((h, w), bool)
    hole[3:-3, 3:-3] = True
    return m, hole


def test_ragged_wall_becomes_straight_with_one_thickness():
    before, hole = room()
    after = before.copy()
    for x in range(3, 97):                                # a wall across the room: wobbles by 1 px, 2 to 5 px thick
        y = 40 + (x // 7) % 2
        after[y:y + 2 + (x // 5) % 4, x] = WALL
    out, info = regularise(before, after, hole)
    added = (out == WALL) & (before != WALL)
    rows = np.unique(np.nonzero(added)[0])
    assert info["walls"] == 1 and len(rows) == 3          # one piece, the plan's wall thickness
    assert added[rows[0], 3:97].all()                     # no gaps, from wall to wall
    c = checks(before, out, hole)
    assert c["thick_sd"] == 0 and c["loose"] == 0 and c["ragged"] == 0


def test_blob_loose_piece_and_curve_are_dropped():
    before, hole = room()
    after = before.copy()
    after[20:32, 50:62] = WALL                            # a blob
    after[50:53, 60:75] = WALL                            # a straight piece that touches nothing
    for k in range(20):                                   # a quarter circle (a door swing read as wall)
        y, x = int(70 - 18 * np.sin(k / 20 * np.pi / 2)), int(45 + 18 * np.cos(k / 20 * np.pi / 2))
        after[y:y + 2, x:x + 2] = WALL
    out, info = regularise(before, after, hole)
    assert info["walls"] == 0
    assert ((out == WALL) == (before == WALL)).all()      # nothing added, nothing removed
    assert (out[hole] == LIV).all()                       # the dropped pixels took the room's type


def test_wall_end_is_extended_to_the_next_wall():
    before, hole = room()
    after = before.copy()
    after[30:33, 44:93] = WALL                            # stops 4 px before the right wall and starts inside the room
    after[3:30, 44:47] = WALL                             # a second wall from its left end up to the top wall
    out, _ = regularise(before, after, hole)
    assert (out[30:33, 93:97] == WALL).all()              # the 4 px gap to the right wall is closed
    assert checks(before, out, hole)["loose"] == 0


def test_closed_new_room_gets_one_type_and_open_space_loses_small_patches():
    before, hole = room()
    after = before.copy()
    after[3:-3, 60:63] = WALL                             # a wall from top to bottom: the right part is a new room
    after[3:-3, 63:-3] = BED
    after[10:30, 70:90] = KIT                             # a patch of another type inside the new room
    after[50:55, 45:50] = KIT                             # a small patch in the open part of the hole
    out, info = regularise(before, after, hole)
    assert info["new_rooms"] == 2                         # both sides of the new wall are closed rooms now
    assert (out[3:-3, 63:-3] == BED).all()
    assert (out[3:-3, 3:60] == LIV).all()


def test_fixture_speck_dropped_fixture_at_wall_kept():
    before, hole = room()
    after = before.copy()
    after[3:9, 50:56] = SINK                              # at the top wall
    after[40:42, 70:72] = SINK                            # a speck in the middle of the room
    out, info = regularise(before, after, hole)
    assert info["fixtures"] == 1
    assert (out[3:9, 50:56] == SINK).all() and (out[40:42, 70:72] == LIV).all()


def test_what_the_fill_removed_stays_removed():
    before, hole = room()
    before[40:42, 50:58] = WALL                           # a stray bit of wall from the parser on erased paper
    after = before.copy()
    after[40:42, 50:58] = LIV
    out, _ = regularise(before, after, hole)
    assert (out[40:42, 50:58] == LIV).all()


def test_one_wall_drawn_twice_becomes_one_wall():
    before, hole = room()
    after = before.copy()
    after[30:33, 3:97] = WALL                             # the same wall twice, 6 px apart; the second is shorter
    after[36:39, 60:97] = WALL
    out, info = regularise(before, after, hole)
    assert info["walls"] == 1
    assert (out[30:33, 3:97] == WALL).all() and (out[36:39, 60:97] != WALL).all()


def test_a_wall_must_end_at_a_wall():
    before, hole = room()
    after = before.copy()
    after[30:33, 3:50] = WALL                             # from the left wall, stops 47 px before the right wall
    after[60:63, 3:83] = WALL                             # stops 14 px before the right wall: a doorway
    after[30:60, 20:23] = WALL                            # joins the two; below the second wall it runs on 8 px
    after[63:73, 20:23] = WALL                            # and stops 4 px before the bottom wall
    out, info = regularise(before, after, hole)
    assert (out[30:33, 3:97] == WALL).all()               # 1: runs on to the right wall
    assert (out[60:63, 83:97] == I["door"]).all() and info["openings"] == 1   # 2: the door-sized gap gets a door
    assert (out[63:77, 20:23] == WALL).all()              # 3: the short gap to the bottom wall is closed with wall
    assert info["open_ends"] == 0

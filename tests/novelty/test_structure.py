"""Round 2 helpers on tiny synthetic maps: wall-continuation rays, ink / stroke maps, wall decoding, region flags."""

import numpy as np

from src.novelty.data import ink_map, stroke_map
from src.novelty.rays import continuation_rays
from src.novelty.structure import decode_walls, targets, wall_clusters
from src.stage2.data.classes16 import C16

WALL, ROOM, DOOR = C16.index["wall"], C16.index["bedroom"], C16.index["door"]


def room():
    """A 60 x 60 room with a wall all round."""
    lab = np.full((60, 60), ROOM, np.uint8)
    lab[:3], lab[-3:], lab[:, :3], lab[:, -3:] = WALL, WALL, WALL, WALL
    return lab


def test_ray_continues_a_wall_that_stops_in_the_open():
    lab = room()
    lab[29:32, 3:25] = WALL                            # a wall from the left side that stops at x = 25
    free, junction = continuation_rays(lab)
    assert free[30, 25:57].all()                       # the ray runs on to the right-hand wall
    assert not free[30, :25].any() and not free[10].any()
    assert not (free & (lab == WALL)).any() and not (free & junction).any()


def test_ray_is_dropped_when_it_crosses_a_door():
    lab = room()
    lab[29:32, 3:25] = WALL
    lab[29:32, 25:35] = DOOR                           # the gap is an opening, not a missing wall
    free, junction = continuation_rays(lab)
    assert not free[30].any() and not junction[30].any()


def test_ink_and_stroke_maps():
    img = np.full((120, 120), 255, np.uint8)
    img[40:44] = 0                                     # a black line, 4 px thick, on a 2x drawing
    ink = ink_map(img, (60, 60))
    assert ink.shape == (60, 60) and ink[20:22].min() > 0.9 * ink.max() and ink[:18].max() == 0
    faint = np.full((60, 60), 250, np.uint8)
    faint[30] = 190                                    # a light grey line: no ink at grey 128, but a clear stroke
    assert not (faint < 128).any()
    st = stroke_map(faint)
    assert st[30].all() and not st[:28].any()


def test_decode_walls_thins_drops_short_pieces_and_skips_existing_walls():
    lab = room()
    heat = np.zeros((60, 60), np.float32)
    heat[28:33, 3:57] = 0.9                            # a 5 px band across the room (touches the side walls)
    heat[10:13, 10:13] = 0.9                           # a speck: shorter than min_len
    add = decode_walls(heat, lab, tau=0.5, min_len=6)
    assert add[30, 10:50].all() and add[:, 30].sum() == 3          # centre line redrawn 3 px wide
    assert not add[8:15, 8:15].any()                               # the speck is gone
    assert not (add & (lab == WALL)).any()                         # never on a wall the map already has
    assert not decode_walls(np.zeros((60, 60), np.float32), lab).any()


def test_wall_clusters_group_nearby_pieces_and_ignore_tiny_ones():
    add = np.zeros((100, 100), bool)
    add[20, 10:40] = True
    add[24, 30:50] = True                              # 4 px away: same group
    add[80, 80:85] = True                              # 5 px in all: below min_px
    boxes = wall_clusters(add, join=8, pad=6, min_px=12)
    assert boxes == [(14, 4, 30, 55)]


def test_targets_mark_only_erased_walls_and_the_area():
    gt = room()
    gt[29:32, 3:57] = WALL
    om = np.zeros((60, 60), np.uint8)
    om[20:40, 10:50] = 1                               # the omission area ...
    om[29:32, 10:50] = 2                               # ... and the erased wall inside it
    band, area = targets(gt, om)
    assert band[28:33, 12:48].all() and band[:27].sum() == 0 and area.sum() == 20 * 40

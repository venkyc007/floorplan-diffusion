"""R8 v2 oversized / merged room -> region missing, relative to the drawing (user, 2026-10-21)."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.detect.merged_rooms import MARGIN, merged_room_candidates

I = C16.index
INF = np.inf


def test_mixed_fixtures_flag_a_merged_room():
    lab = np.full((80, 80), I["background"], np.uint8)
    lab[10:70, 10:70] = I["bathroom"]
    lab[12:18, 12:18] = I["toilet"]
    lab[60:68, 60:68] = I["appliance"]                     # a kitchen appliance in the "bathroom"
    c, = merged_room_candidates(lab, INF, INF, mix=True)
    assert c.type == "region" and c.rules == ["R8"] and c.mask[40, 40]
    assert c.box == (10 - MARGIN, 10 - MARGIN, 69 + MARGIN, 69 + MARGIN)
    assert merged_room_candidates(lab, INF, INF, mix=False) == []
    lab[60:68, 60:68] = I["bathroom"]                      # toilet only: one room type takes it
    assert merged_room_candidates(lab, INF, INF, mix=True) == []


def test_room_twice_the_size_of_its_siblings():
    lab = np.full((60, 200), I["wall"], np.uint8)
    for x0 in (2, 32, 62):                                 # three 26 x 26 bedrooms
        lab[2:28, x0:x0 + 26] = I["bedroom"]
    lab[2:58, 92:118] = I["bedroom"]                       # a bedroom ~2.2x the others: merged with a neighbour
    c = merged_room_candidates(lab, 1.8, 2.5, mix=False)
    assert len(c) == 1 and c[0].mask[40, 100]
    assert merged_room_candidates(lab, 2.4, 2.5, mix=False) == []


def test_single_type_compared_with_all_rooms():
    lab = np.full((60, 200), I["wall"], np.uint8)
    for x0 in (2, 32, 62):
        lab[2:28, x0:x0 + 26] = I["bedroom"]
    lab[2:58, 92:150] = I["kitchen"]                       # the only kitchen, ~4.8x the median room
    c = merged_room_candidates(lab, 1.8, 2.5, mix=False)
    assert len(c) == 1 and c[0].mask[30, 120]
    assert merged_room_candidates(lab, 1.8, 5.0, mix=False) == []

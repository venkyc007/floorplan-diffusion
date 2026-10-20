"""R8 oversized / merged room -> region missing (user, 2026-10-20)."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.detect.merged_rooms import P99, merged_room_candidates

I = C16.index


def test_mixed_fixtures_flag_a_merged_room():
    lab = np.full((80, 80), I["background"], np.uint8)
    lab[10:70, 10:70] = I["bathroom"]
    lab[12:18, 12:18] = I["toilet"]
    lab[60:68, 60:68] = I["appliance"]                     # a kitchen appliance in the "bathroom"
    c, = merged_room_candidates(lab, area_mult=np.inf, mix=True)
    assert c.type == "region" and c.rules == ["R8"] and c.mask[40, 40]
    assert merged_room_candidates(lab, area_mult=np.inf, mix=False) == []
    lab[60:68, 60:68] = I["bathroom"]                      # toilet only: one room type takes it
    assert merged_room_candidates(lab, area_mult=np.inf, mix=True) == []


def test_oversized_room():
    side = int(np.ceil(np.sqrt(P99["bathroom"] * 1.3)))
    lab = np.full((side + 4, side + 4), I["background"], np.uint8)
    lab[2:side + 2, 2:side + 2] = I["bathroom"]
    assert len(merged_room_candidates(lab, area_mult=1.25, mix=False)) == 1
    assert merged_room_candidates(lab, area_mult=1.5, mix=False) == []

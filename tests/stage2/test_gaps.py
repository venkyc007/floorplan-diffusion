"""Doorless-gap rule R5: fires for a bedroom left open to a hallway, not for open-plan pairs or closets."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.detect.gaps import gap_candidates

I = C16.index
KW = {"need_area": 500, "small_other": 300, "min_len": 8}


def plan(top, bottom, gap=12, bottom_rows=(33, 60)):
    """Two spaces split by a horizontal 3 px wall (rows 30-32) with a gap of `gap` px at x 30.."""
    lab = np.full((64, 80), I["background"], np.uint8)
    lab[2:30, 2:78] = I[top]
    lab[bottom_rows[0]:bottom_rows[1], 2:78] = I[bottom]
    lab[30:33, 2:78] = I["wall"]
    lab[30:33, 30:30 + gap] = I[top]                       # the gap: filled by the top space
    lab[0:2, :] = lab[:, 0:2] = lab[:, 78:] = I["wall"]
    return lab


def test_bedroom_gap_fires():
    c = gap_candidates(plan("bedroom", "hallway"), **KW)
    assert len(c) == 1 and c[0].rules == ["R5"] and c[0].type == "door"


def test_open_plan_pair_ignored():
    assert gap_candidates(plan("living", "kitchen"), **KW) == []


def test_closet_ignored_and_short_gap_ignored():
    assert gap_candidates(plan("bedroom", "other_room", bottom_rows=(33, 36)), **KW) == []   # 228 px closet
    assert gap_candidates(plan("bedroom", "hallway", gap=6), **KW) == []

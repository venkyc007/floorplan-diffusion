"""Unit tests for src/stage1/postprocess.py. Run:  python -m pytest tests/"""

import numpy as np

from src.stage1.data.classes import ROOM, WALL
from src.stage1.eval.arch_metrics import dead_ends
from src.stage1.postprocess import clean
from tests.stage1.test_metrics import toy_plan

EVERYWHERE = np.ones((40, 40), bool)


def test_short_stub_is_removed():
    plan = toy_plan()
    plan[24:26, 12:15] = WALL                      # a 3-px free-standing stub in the left room
    fixed = clean(plan, EVERYWHERE, n=6)
    assert (fixed[24:26, 12:15] == ROOM).all()
    assert dead_ends(fixed, EVERYWHERE) == 0


def test_stub_close_to_a_wall_is_extended():
    plan = toy_plan()
    plan[28:30, 8:17] = WALL                       # a wall (away from the door at rows 18-21) ...
    # ... ending 2 px short of the interior wall at cols 19-20: gap at cols 17-18
    fixed = clean(plan, EVERYWHERE, n=6)
    assert (fixed[28:30, 17:19] == WALL).all()     # gap closed


def test_correct_plan_is_unchanged():
    plan = toy_plan()
    assert (clean(plan, EVERYWHERE, n=8) == plan).all()


def test_pixels_outside_the_mask_never_change():
    plan = toy_plan()
    plan[24:26, 12:15] = WALL
    mask = np.zeros((40, 40), bool)                # stub entirely OUTSIDE the mask
    mask[8:12, 8:12] = True
    assert (clean(plan, mask, n=6) == plan).all()

"""Tests of the app helpers added for the combined Stage 1 + Stage 2 interface (no model is loaded)."""

import numpy as np
import pytest

from src.stage1.data.corruptions import UNKNOWN
from src.stage1.interface.random_mask import random_corruption
from src.stage2.interface.demo import InputError, mask_boxes, to_grey
from tests.stage1.test_metrics import toy_plan


def test_to_grey_handles_colour_alpha_and_dark_background():
    rgb = np.full((100, 120, 3), 255, np.uint8)
    rgb[40:60, 10:110] = 0                                     # a dark line on white paper
    g = to_grey(rgb)
    assert g.shape == (100, 120) and g.dtype == np.uint8 and g[50, 50] == 0 and g[5, 5] == 255
    rgba = np.zeros((100, 120, 4), np.uint8)                   # fully transparent = paper
    rgba[40:60, 10:110] = (0, 0, 0, 255)
    g = to_grey(rgba)
    assert g[5, 5] == 255 and g[50, 50] == 0
    assert to_grey(255 - rgb)[50, 50] == 0                     # white lines on black are inverted


def test_to_grey_refuses_tiny_images():
    with pytest.raises(InputError):
        to_grey(np.full((20, 20), 255, np.uint8))


def test_mask_boxes_one_box_per_region_and_stray_clicks_ignored():
    m = np.zeros((200, 200), bool)
    m[10:40, 20:60] = True
    m[100:150, 120:130] = True
    m[190, 190] = True                                         # a stray click
    boxes = sorted(mask_boxes(m))
    assert len(boxes) == 2
    assert boxes[0][0] <= 10 and boxes[0][2] >= 39 and boxes[1][1] <= 120 and boxes[1][3] >= 129


def test_random_corruption_keeps_the_truth_and_masks_only_unknown():
    plan = toy_plan()
    out = random_corruption(plan, seed=3)
    assert out is not None
    corrupted, mask, truth, text = out
    assert (truth == plan).all() and mask.any()
    assert ((corrupted == UNKNOWN) == mask).all()              # unknown exactly inside the mask
    assert (corrupted[~mask] == plan[~mask]).all()
    with pytest.raises(ValueError):
        random_corruption(corrupted, seed=3)                   # a plan with unknown pixels is not complete

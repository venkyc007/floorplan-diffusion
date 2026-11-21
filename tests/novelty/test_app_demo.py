"""Tests of the app's novelty tab helpers (src/novelty/app_demo.py); no model is loaded."""

import numpy as np
import pytest

from src.novelty import app_demo as NV
from src.novelty import pipeline as nov
from src.stage2.data.classes16 import C16

WALL, BED, BG = C16.index["wall"], C16.index["bedroom"], C16.index["background"]


def test_all_versions_share_the_trained_models():
    """The app loads the models once (pipeline.models is cached), so every version must name the same checkpoints."""
    ck = [NV.checkpoints(p) for p in NV.VERSIONS.values()]
    assert len(ck) == 3 and all(c == ck[0] for c in ck)


def test_use_version_points_the_pipeline_at_the_frozen_config():
    old = nov.CFG_FILE
    try:
        for label, path in NV.VERSIONS.items():
            NV.use_version(label)
            assert nov.CFG_FILE.as_posix().endswith(path)
        with pytest.raises(ValueError):
            NV.use_version("v9 · does not exist")
    finally:
        nov.CFG_FILE = old


def toy_plan():
    p = np.full((80, 80), BG, np.uint8)
    p[10:70, 10:70] = BED
    p[10:13, 10:70] = WALL
    p[67:70, 10:70] = WALL
    p[10:70, 10:13] = WALL
    p[10:70, 67:70] = WALL
    return p


def test_problems_image_marks_a_wall_ending_in_the_air():
    plan = toy_plan()
    region = np.zeros(plan.shape, bool)
    region[5:75, 5:75] = True
    clean, found = NV.problems_image(plan, region)
    assert not found["dead_ends"]
    plan[30:33, 13:45] = WALL                              # a wall from the left wall that stops in the room
    img, found = NV.problems_image(plan, region)
    assert found["dead_ends"] and img.shape == clean.shape and (img != clean).any()


def test_problems_image_without_region_flags_is_just_the_plan():
    img, found = NV.problems_image(toy_plan(), np.zeros((80, 80), bool))
    assert found == {} and img.ndim == 3


def test_compare_image_puts_the_versions_side_by_side():
    tile = np.full((100, 120, 3), 200, np.uint8)
    steps = {"problems": tile, "sound": {"sound": True}}
    out = NV.compare_image([("v4c · a", steps), ("v4b · b", steps), ("v3c · c", {**steps, "sound": None})])
    assert out.ndim == 3 and out.shape[1] > 3 * 100

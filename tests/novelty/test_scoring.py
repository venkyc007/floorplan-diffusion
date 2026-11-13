"""Novelty scoring and pipeline helpers on tiny synthetic maps."""

import numpy as np

from src.novelty.eval_maps import wall_recovery
from src.novelty.pipeline import outline_walls
from src.stage2.data.classes16 import C16

WALL, ROOM = C16.index["wall"], C16.index["bedroom"]


def maps():
    """Truth: a room split by one vertical wall at x = 30..32. Parsed (erased): the wall is gone."""
    gt = np.full((60, 60), ROOM, np.uint8)
    gt[:, 30:33] = WALL
    parsed = np.full((60, 60), ROOM, np.uint8)
    rm = gt == WALL                                   # the erased structure = that wall
    region = np.ones((60, 60), bool)
    return gt, parsed, rm, region


def test_wall_recovery_perfect_shifted_and_wrong():
    gt, parsed, rm, region = maps()
    assert wall_recovery(gt, parsed, gt, rm, region) == {"miss_recall": 1.0, "add_precision": 1.0, "recovered": True}
    shifted = parsed.copy()
    shifted[:, 32:35] = WALL                          # the wall drawn 2 px off: strict IoU would be 1/5
    r = wall_recovery(shifted, parsed, gt, rm, region)
    assert r["recovered"] and r["miss_recall"] == 1.0 and r["add_precision"] == 1.0
    wrong = parsed.copy()
    wrong[30:33, :] = WALL                            # a horizontal wall instead: little recall, little precision
    r = wall_recovery(wrong, parsed, gt, rm, region)
    assert not r["recovered"] and r["miss_recall"] < 0.2 and r["add_precision"] < 0.2
    r = wall_recovery(parsed, parsed, gt, rm, region)  # nothing drawn
    assert not r["recovered"] and r["miss_recall"] == 0.0 and r["add_precision"] == ""


def test_wall_recovery_ignores_walls_the_parse_already_has():
    gt, parsed, rm, region = maps()
    parsed[:, 30:33] = WALL                           # the parser still shows the wall: nothing is truly missing
    r = wall_recovery(parsed, parsed, gt, rm, region)
    assert r["miss_recall"] == "" and not r["recovered"]


def test_outline_walls_paints_only_inside_the_hole_and_away_from_walls():
    sketch = np.full((40, 40), ROOM, np.uint8)
    sketch[10, 10:30] = WALL                          # the sketch already has the top wall
    blob = np.zeros((40, 40), bool)
    blob[10:30, 10:30] = True
    hole = np.zeros((40, 40), bool)
    hole[5:35, 5:25] = True                           # the hole covers only the left part of the blob
    out = outline_walls(sketch, blob, hole)
    added = (out == WALL) & (sketch != WALL)
    assert added.any() and not added[~hole].any()     # nothing outside the hole
    assert not added[10:13, 12:22].any()              # no second wall next to the existing top wall
    assert added[15:25, 10:13].all()                  # the blob's left edge got its wall band

"""Door clipping (an added door stays on one wall line at median length) and the per-drawing door-ink guard."""

import numpy as np

from src.stage2.detect.door_ink import guarded
from src.stage2.pipeline.run import clip_door


def test_l_shaped_door_keeps_one_arm_at_median_length():
    door = np.zeros((60, 60), bool)
    door[20:23, 10:45] = True          # long horizontal arm, 3 px thick
    door[20:40, 42:45] = True          # vertical arm on another wall
    out = clip_door(door, length=13)
    ys, xs = np.nonzero(out)
    assert set(ys) <= {20, 21, 22}     # only the horizontal wall line
    assert xs.max() - xs.min() + 1 <= 14


def test_ink_guard():
    assert guarded([0.0, 0.1, 0.2, 5.0], 1.5)          # 3 of 4 doors without ink: skip the drawing
    assert not guarded([0.0, 4.0, 6.0, 8.0], 1.5)      # one door without ink: keep the candidate
    assert not guarded([], 1.5)

"""Unit tests for src/stage1/eval/arch_metrics.py. Run:  python -m pytest tests/

Each rule must pass the hand-made toy plan (correct) and fail a copy with
its specific defect. The toy plan (tests/stage1/test_metrics.py) has two rooms,
2-px walls, an interior door at rows 18-21 / cols 19-20, a window in the left
outer wall at rows 10-15 / cols 4-5 and an outdoor strip on the right.
"""

import numpy as np

from src.stage1.data.classes import BACKGROUND, DOOR, ROOM, WALL, WINDOW
from src.stage1.eval.arch_metrics import RULES, case_metrics, label_metrics, room_shapes
from tests.stage1.test_metrics import toy_plan

EVERYWHERE = np.ones((40, 40), bool)
TH = {"min_area": 72.0, "min_width": 7.64}


def m(label):
    return label_metrics(label, EVERYWHERE, TH)


def test_correct_plan_passes_every_rule():
    out = m(toy_plan())
    for rule in RULES + ["opening_fit", "slivers"]:
        assert out[rule] == 0, (rule, out[rule])
    assert out["n_openings"] == 2


def test_dead_end_wall():
    plan = toy_plan()
    plan[24:26, 7:15] = WALL                       # a wall stub ending in the middle of a room
    assert m(plan)["dead_ends"] >= 1


def test_floating_fragment():
    plan = toy_plan()
    plan[26:28, 10:12] = WALL                      # a 4-px wall blob, not touching anything
    assert m(plan)["fragments"] == 1


def test_opening_fit_floating_door_and_wrong_thickness():
    floating = toy_plan()
    floating[18:22, 19:21] = WALL                  # close the real door ...
    floating[25:27, 8:14] = DOOR                   # ... and float one in the room
    out = m(floating)
    assert out["opening_bridge"] == 1 and out["opening_fit"] == 1
    thick = toy_plan()
    thick[16:25, 17:23] = DOOR                     # 9 px long, 6 px thick door in a 2-px wall
    out = m(thick)
    assert out["opening_fit"] == 1 and out["opening_bridge"] == 0   # it bridges, but doesn't fit


def test_door_must_connect_two_spaces():
    plan = toy_plan()
    plan[18:22, 21] = WALL                         # wall right behind the door on one side
    assert m(plan)["door_spaces"] == 1


def test_window_must_face_outside():
    plan = toy_plan()
    plan[26:30, 19:21] = WINDOW                    # a window in the INTERIOR wall: rooms on both sides
    assert m(plan)["window_out"] == 1


def test_room_reachability():
    plan = toy_plan()
    plan[18:22, 19:21] = WALL                      # remove the only door between the two rooms
    assert m(plan)["unreachable"] == 1


def test_sliver_room():
    plan = toy_plan()
    plan[6:34, 6:19] = WALL                        # fill the left room with wall ...
    plan[19:21, 6:19] = ROOM                       # ... except a 2-px wide corridor
    assert m(plan)["slivers"] >= 1


def test_rectangularity():
    rect = room_shapes(toy_plan(), EVERYWHERE)
    plan = toy_plan()
    plan[6:20, 6:12] = WALL                        # make the left room L-shaped
    l_shape = room_shapes(plan, EVERYWHERE)
    assert min(r for _, _, r in rect) > 0.99
    assert min(r for _, _, r in l_shape) < 0.9


def test_validity_compares_with_ground_truth():
    gt = toy_plan()
    mask = np.zeros((40, 40), bool)
    mask[22:32, 6:18] = True
    assert case_metrics(gt.copy(), gt, mask, TH)["valid"]
    bad = gt.copy()
    bad[26:28, 10:12] = WALL                       # a floating fragment inside the mask
    out = case_metrics(bad, gt, mask, TH)
    assert out["pred_fragments"] == 1 and out["gt_fragments"] == 0 and not out["valid"]

"""Unit tests for src/eval/metrics.py. Run:  python -m pytest tests/

They use a tiny hand-drawn plan (below) so every expected number can be
worked out by hand.
"""

import numpy as np
import pytest

from src.data.classes import BACKGROUND, DOOR, OUTDOOR, ROOM, WALL, WINDOW
from src.eval.metrics import enclosed_regions, sample_stats, summarize


def toy_plan():
    """A 40x40 plan: two rooms side by side, 2-px walls, one interior door,
    one window in the outer wall, an outdoor strip on the right.

        cols  4..35 building, rows 4..35
        outer wall: 2 px thick ring; interior wall at cols 19-20
        door:   rows 18-21 of the interior wall (cols 19-20) -> 8 px
        window: rows 10-15 of the left outer wall (cols 4-5) -> 12 px
        outdoor: cols 36-39
    """
    p = np.full((40, 40), BACKGROUND, np.uint8)
    p[4:36, 4:36] = WALL
    p[6:34, 6:19] = ROOM          # left room
    p[6:34, 21:34] = ROOM         # right room
    p[18:22, 19:21] = DOOR        # interior door
    p[10:16, 4:6] = WINDOW        # window in left outer wall
    p[:, 36:40] = OUTDOOR
    return p


def door_mask():
    """Mask = door bounding box (rows 18-21, cols 19-20) + 3 px each side."""
    m = np.zeros((40, 40), bool)
    m[15:25, 16:24] = True
    return m


def test_perfect_prediction():
    gt, mask = toy_plan(), door_mask()
    s = summarize([sample_stats(gt.copy(), gt, mask)])
    assert s["mIoU"] == 1.0 and s["tmIoU"] == 1.0
    for name in ("wall", "door", "room"):
        assert s[f"IoU_{name}"] == 1.0
    assert s["tIoU_wall"] == 1.0 and s["tIoU_door"] == 1.0
    assert np.isnan(s["IoU_window"])       # not in the mask -> no score
    assert s["pixel_acc"] == 1.0
    assert s["F1_door"] == 1.0 and s["F1_room"] == 1.0
    assert s["conn_err"] == 0 and s["room_count_err"] == 0
    assert s["open_on_wall"] == 1.0        # the door touches walls
    assert s["room_leak"] == 0.0           # rooms are enclosed


def test_empty_prediction_scores_poorly():
    gt, mask = toy_plan(), door_mask()
    empty = np.full_like(gt, BACKGROUND)   # model draws nothing in the mask
    s = summarize([sample_stats(empty, gt, mask)])
    assert s["mIoU"] == 0.0 and s["tmIoU"] == 0.0
    assert s["tIoU_wall"] == 0.0 and s["tIoU_door"] == 0.0
    assert s["pixel_acc"] == 0.0
    assert s["R_door"] == 0.0 and s["F1_door"] == 0.0
    # The background hole cuts both rooms open -> they now touch background.
    assert s["room_leak"] == 1.0
    # Both halves of the cut interior wall still connect through the outer
    # wall ring, so the boundary stays ONE piece: conn_err does not see this
    # hole (room_leak does, above).
    assert s["conn_err"] == 0
    # ...but the hole joins the two rooms into one enclosed region (was 2).
    assert s["room_count_err"] == 1


def test_enclosed_regions_toy_plan():
    """Two rooms -> 2 enclosed regions; the outdoor strip and background
    touch the image edge, so they are outside and not counted."""
    assert enclosed_regions(toy_plan()) == 2


def test_known_answer_door_filled_with_wall():
    """The model draws wall where the door was, everything else right."""
    gt, mask = toy_plan(), door_mask()
    pred = gt.copy()
    pred[gt == DOOR] = WALL
    s = sample_stats(pred, gt, mask)
    # Inside the mask (10 x 8 = 80 px): door 8 px, wall 2 cols x 6 rows
    # above/below the door = 12 px... computed directly from the plan:
    wall_in_mask = int(((gt == WALL) & mask).sum())
    assert s["inter_wall"] == wall_in_mask
    assert s["union_wall"] == wall_in_mask + 8           # + the 8 former door px
    assert s["inter_door"] == 0 and s["union_door"] == 8
    assert s["miou"] == pytest.approx(np.mean([wall_in_mask / (wall_in_mask + 8), 0.0, 1.0]))
    assert (s["tp_door"], s["fp_door"], s["fn_door"]) == (0, 0, 1)
    assert s["correct_px"] == 80 - 8
    # Door -> wall: still boundary, so the boundary stays one piece and the
    # two rooms stay two enclosed regions.
    assert s["conn_err"] == 0 and s["room_count_err"] == 0
    # Tolerant IoU: every predicted wall pixel (incl. the former door) is
    # within 1 px of a true wall pixel, but the door is simply missing.
    assert s["tmatch_door"] == 0


def test_floating_door_is_not_on_wall():
    """A door drawn in the middle of a room does not touch any wall."""
    gt, mask = toy_plan(), np.zeros((40, 40), bool)
    mask[8:16, 8:16] = True
    pred = gt.copy()
    pred[11:13, 11:13] = DOOR
    s = sample_stats(pred, gt, mask)
    assert s["n_openings"] == 1 and s["n_openings_on_wall"] == 0
    assert (s["tp_door"], s["fp_door"], s["fn_door"]) == (0, 1, 0)
    # The floating door is a separate piece of boundary: one extra component.
    assert s["conn_err"] == 1


def test_alignment_straight_vs_diagonal():
    """Straight walls score 1; a diagonal staircase wall scores lower."""
    gt, mask = toy_plan(), np.zeros((40, 40), bool)
    mask[6:34, 6:19] = True                       # inside the left room
    straight = gt.copy()
    straight[20:22, 6:19] = WALL                  # a new straight wall
    diagonal = gt.copy()
    for k in range(12):                           # a 1-px diagonal line
        diagonal[8 + k, 7 + k] = WALL
    s_straight = sample_stats(straight, gt, mask)
    s_diag = sample_stats(diagonal, gt, mask)
    assert s_straight["aligned_px"] == s_straight["boundary_px"]
    assert s_diag["aligned_px"] / s_diag["boundary_px"] < 0.5


def test_tolerant_iou_forgives_one_pixel_shift():
    """A wall drawn 1 px off: strict IoU drops, tolerant IoU stays 1."""
    gt, mask = toy_plan(), np.zeros((40, 40), bool)
    mask[6:34, 6:19] = True
    gt = gt.copy()
    gt[20:22, 6:19] = WALL                         # true wall in the left room
    pred = gt.copy()
    pred[20:22, 6:19] = ROOM
    pred[21:23, 6:19] = WALL                       # same wall, 1 px lower
    s = sample_stats(pred, gt, mask)
    assert s["inter_wall"] / s["union_wall"] < 0.5
    assert s["tmatch_wall"] / (s["ttotal_wall"] - s["tmatch_wall"]) == 1.0

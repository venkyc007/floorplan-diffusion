"""Unit tests for src/stage1/models/constraints.py. Run:  python -m pytest tests/

Each test builds a hand-made plan (the 40x40 toy plan from test_metrics:
two rooms, 2-px walls, an interior door, a window, an outdoor strip),
breaks one thing, and checks that the matching loss gets worse.
"""

import torch

from src.stage1.data.classes import BACKGROUND, DOOR, NUM_CLASSES, ROOM, WALL
from src.stage1.models.constraints import (alignment_loss, cldice_loss, closure_loss, constraint_losses,
                                    grow, noise_weight, openings_loss, probabilities)
from tests.stage1.test_metrics import toy_plan

TEMPERATURE = 0.2


def as_x0(label):
    """Label map (H, W) -> predicted-x0-like tensor (1, 6, H, W) in [-1, 1]."""
    onehot = torch.nn.functional.one_hot(torch.as_tensor(label).long(), NUM_CLASSES)
    return onehot.permute(2, 0, 1)[None].float() * 2 - 1


def probs(label):
    return probabilities(as_x0(label), TEMPERATURE)


def target_of(label):
    return (as_x0(label) + 1) / 2


EVERYWHERE = torch.ones(1, 1, 40, 40)


def test_broken_wall_scores_worse_than_connected():
    plan = toy_plan()
    broken = plan.copy()
    broken[8:14, 19:21] = ROOM                      # cut the interior wall
    target = target_of(plan)
    good = cldice_loss(probs(plan), target, EVERYWHERE, iterations=4)
    bad = cldice_loss(probs(broken), target, EVERYWHERE, iterations=4)
    assert good.item() < 0.02
    assert bad.item() > good.item() + 0.01


def test_floating_door_scores_worse_than_door_in_wall():
    plan = toy_plan()                               # its door sits in the interior wall
    floating = plan.copy()
    floating[18:22, 19:21] = WALL                   # close the wall where the door was
    floating[24:26, 8:14] = DOOR                    # a door in the middle of the left room
    in_wall = openings_loss(probs(plan), EVERYWHERE, iterations=12)
    loose = openings_loss(probs(floating), EVERYWHERE, iterations=12)
    assert in_wall.item() < 0.0005                  # also the door's middle pixels are supported
    assert loose.item() > 10 * max(in_wall.item(), 1e-4)


def test_leaking_room_scores_worse_than_closed_room():
    plan = toy_plan()
    leaking = plan.copy()
    leaking[4:6, 8:14] = ROOM                       # hole in the outer wall, room spills out
    leaking[0:4, 8:14] = BACKGROUND
    closed = closure_loss(probs(plan), EVERYWHERE)
    leak = closure_loss(probs(leaking), EVERYWHERE)
    assert closed.item() < 0.0005
    assert leak.item() > 10 * max(closed.item(), 1e-4)


def test_diagonal_wall_scores_worse_than_straight_wall():
    plan = toy_plan()
    straight, diagonal = plan.copy(), plan.copy()
    straight[25:27, 7:18] = WALL                    # an extra horizontal wall
    for k in range(10):                             # an extra 2-px diagonal wall
        diagonal[22 + k, 7 + k:9 + k] = WALL
    region = torch.zeros(1, 1, 40, 40)
    region[..., 20:34, 6:19] = 1                    # judge only the left room
    assert alignment_loss(probs(diagonal), region).item() > alignment_loss(probs(straight), region).item() + 0.1


def test_losses_only_look_near_the_mask():
    """A broken wall far from the (grown) mask must not change the loss."""
    plan = toy_plan()
    broken = plan.copy()
    broken[8:14, 19:21] = ROOM
    mask = torch.zeros(1, 1, 40, 40)
    mask[..., 28:32, 8:12] = 1                      # far away from the cut
    cfg = {"temperature": TEMPERATURE, "dilate_px": 3, "t_max": 600,
           "cldice": {"use": True, "iterations": 4}, "openings": {"use": True, "iterations": 12},
           "closure": {"use": True}, "alignment": {"use": True}}
    a = constraint_losses(as_x0(plan), target_of(plan), mask, cfg)
    b = constraint_losses(as_x0(broken), target_of(plan), mask, cfg)
    for name in a:
        assert abs(a[name].item() - b[name].item()) < 1e-6, name
    assert grow(mask, 3).sum() == (4 + 6) ** 2


def test_no_opening_no_penalty():
    """A region with no door/window and no room must not be penalised for the
    tiny leftover probabilities the softmax leaves on every class."""
    plan = toy_plan()
    region = torch.zeros(1, 1, 40, 40)
    region[..., 24:30, 9:15] = 1                    # inside the left room: no opening here
    assert openings_loss(probs(plan), region, iterations=12).item() < 1e-3
    outdoor_only = torch.zeros(1, 1, 40, 40)
    outdoor_only[..., 5:30, 37:40] = 1               # inside the outdoor strip: no room here
    assert closure_loss(probs(plan), outdoor_only).item() < 1e-3


def test_noise_weight_cuts_off_high_noise():
    w = noise_weight(torch.tensor([0, 599, 600, 999]), {"t_max": 600})
    assert w.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_losses_are_differentiable():
    plan = toy_plan()
    x0 = as_x0(plan).requires_grad_(True)
    cfg = {"temperature": TEMPERATURE, "dilate_px": 3, "t_max": 600,
           "cldice": {"use": True, "iterations": 4}, "openings": {"use": True, "iterations": 12},
           "closure": {"use": True}, "alignment": {"use": True}}
    total = sum(v.sum() for v in constraint_losses(x0 * 0.5, target_of(plan), EVERYWHERE, cfg).values())
    total.backward()
    assert x0.grad is not None and torch.isfinite(x0.grad).all() and x0.grad.abs().sum() > 0


CFG = {"temperature": TEMPERATURE, "dilate_px": 3, "t_min": 0, "t_max": 1000,
       "cldice": {"use": True, "iterations": 4}, "openings": {"use": True, "iterations": 12},
       "closure": {"use": True}, "alignment": {"use": True}}


def test_perfect_prediction_costs_nothing():
    """Ground-truth-relative losses: predicting the true plan scores ~0 on
    every constraint (real corners / real exterior doors are not penalised)."""
    plan = toy_plan()
    out = constraint_losses(as_x0(plan), target_of(plan), EVERYWHERE, CFG)
    for name, value in out.items():
        assert value.item() < 1e-3, (name, value.item())


def test_no_reward_for_deleting_openings():
    """Removing the real door (wall drawn instead) must not score BETTER than
    the true plan on the openings rule (no cheating by deleting openings)."""
    plan = toy_plan()
    no_door = plan.copy()
    no_door[18:22, 19:21] = WALL
    target = target_of(plan)
    honest = constraint_losses(as_x0(plan), target, EVERYWHERE, CFG)["openings"]
    cheat = constraint_losses(as_x0(no_door), target, EVERYWHERE, CFG)["openings"]
    assert cheat.item() >= honest.item() - 1e-6


def test_relative_losses_still_catch_errors():
    """Relative to the ground truth, a floating door and a leaking room are still penalised."""
    plan = toy_plan()
    target = target_of(plan)
    floating = plan.copy()
    floating[18:22, 19:21] = WALL
    floating[24:26, 8:14] = DOOR
    leaking = plan.copy()
    leaking[4:6, 8:14] = ROOM
    leaking[0:4, 8:14] = BACKGROUND
    assert constraint_losses(as_x0(floating), target, EVERYWHERE, CFG)["openings"].item() > 1e-3
    assert constraint_losses(as_x0(leaking), target, EVERYWHERE, CFG)["closure"].item() > 1e-3

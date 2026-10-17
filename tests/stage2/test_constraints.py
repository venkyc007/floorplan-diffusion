"""Gate G1: every constraint is ~0 on a valid toy plan and > 0 on its broken toy.

Also checks the operators (shift, even-sized openings, flank, flood fill), the
check-only rules R1-R4, and that every loss gives finite gradients on soft inputs.
Run: python -m pytest tests/stage2/
"""

import numpy as np
import pytest
import torch

from src.stage2.constraints import ops, toys
from src.stage2.constraints.constraints import REGISTRY, Ctx, constraint_loss, one_hot, violation_map
from src.stage2.constraints.rules import r1_room_area, r2_room_count, r3_adjacency, r4_exterior_door

NAMES = list(REGISTRY)
# Constraints whose broken toy is tested in "inference" form (no ground truth, no hole).
NO_GT = {"C1", "C8"}


def P_of(label):
    return one_hot(torch.from_numpy(label)[None])


def ctx_for(name, hole):
    if name in NO_GT:
        return Ctx(const=toys.TOY_CONST, K=200)
    h = None if hole is None else torch.from_numpy(hole).float()[None, None]
    return Ctx(hole=h, Y=P_of(toys.valid_plan()), const=toys.TOY_CONST, K=200)


@pytest.mark.parametrize("name", NAMES)
def test_valid_toy_is_clean(name):
    _, hole = toys.broken(name)
    v = violation_map(name, P_of(toys.valid_plan()), ctx_for(name, hole))
    assert v.shape == (1, 1, toys.SIZE, toys.SIZE)
    assert float(v.max()) < 1e-4, f"{name} fires on the valid plan"


@pytest.mark.parametrize("name", NAMES)
def test_broken_toy_violates(name):
    lab, hole = toys.broken(name)
    v = violation_map(name, P_of(lab), ctx_for(name, hole))
    assert float(v.sum()) > 0.5, f"{name} misses its broken plan"


@pytest.mark.parametrize("name", NAMES)
def test_loss_gradients_finite(name):
    """Soft probabilities (noisy logits) -> finite loss and finite, non-NaN gradients."""
    torch.manual_seed(0)
    lab, hole = toys.broken(name)
    logits = (P_of(lab) * 6 + torch.randn(1, 16, toys.SIZE, toys.SIZE)).requires_grad_()
    P = torch.softmax(logits, 1)
    h = torch.from_numpy(toys.HOLES.get(name, toys.hole(20, 70, 20, 70))).float()[None, None]
    ctx = Ctx(hole=h, Y=P_of(toys.valid_plan()), const=toys.TOY_CONST, K=64)
    loss = constraint_loss(name, P, ctx).sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("name", [n for n in NAMES if REGISTRY[n].mode == "gt_hinged"])
def test_hinged_loss_zero_on_ground_truth(name):
    _, hole = toys.broken(name)
    h = torch.from_numpy(toys.hole(10, 85, 10, 85) if hole is None else hole).float()[None, None]
    Y = P_of(toys.valid_plan())
    ctx = Ctx(hole=h, Y=Y, const=toys.TOY_CONST, K=200)
    assert float(constraint_loss(name, Y, ctx)) == pytest.approx(0.0, abs=1e-6)


def test_cldice_loss_zero_on_ground_truth_and_positive_on_broken():
    h = torch.from_numpy(toys.hole(40, 90, 20, 45)).float()[None, None]
    Y = P_of(toys.valid_plan())
    ctx = Ctx(hole=h, Y=Y, const=toys.TOY_CONST)
    assert float(constraint_loss("C1", Y, ctx)) < 1e-4
    assert float(constraint_loss("C1", P_of(toys.broken("C1")[0]), ctx)) > 0.01


# ------------------------------------------------------------------ check-only rules

@pytest.mark.parametrize("rule,fn", [("R1", r1_room_area), ("R3", r3_adjacency), ("R4", r4_exterior_door)])
def test_rules(rule, fn):
    assert not fn(toys.valid_plan(), toys.TOY_CONST)[0]
    bad, mask = fn(toys.broken(rule)[0], toys.TOY_CONST)
    assert bad and mask.any()


def test_r2_room_count():
    valid = toys.valid_plan()
    hole = toys.hole(15, 40, 35, 60)
    assert not r2_room_count(valid, valid, hole)[0]
    changed = valid.copy()
    toys.box(changed, 49, 84, 11, 29, "wall")   # bedroom removed, far from this hole
    assert r2_room_count(changed, valid, hole)[0]


# ------------------------------------------------------------------ operators

def test_shift():
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 2, 2] = 1
    assert ops.shift(x, 1, -2)[0, 0, 3, 0] == 1
    assert ops.shift(x, 5, 0).sum() == 0


@pytest.mark.parametrize("n", [4, 5, 8, 9])
def test_opening_keeps_long_runs_exactly(n):
    """Even and odd line openings keep runs >= n unchanged and delete shorter ones."""
    x = torch.zeros(1, 1, 3, 40)
    x[0, 0, 1, 5:5 + n] = 1          # exactly n long: kept
    x[0, 0, 1, 25:25 + n - 1] = 1    # n - 1 long: removed
    y = ops.opn_h(x, n)
    assert torch.equal(y[0, 0, 1, 5:5 + n], torch.ones(n))
    assert y[0, 0, 1, 20:].sum() == 0


def test_erosion_does_not_eat_from_the_border():
    x = torch.ones(1, 1, 10, 10)
    assert torch.equal(ops.ero(x, 5), x)


def test_flank():
    x = torch.zeros(1, 1, 1, 20)
    x[0, 0, 0, 2] = x[0, 0, 0, 12] = 1
    f = ops.flank_h(x, 6)[0, 0, 0]
    assert f[7] == 1 and f[3] == 0 and f[18] == 0


def test_rec_floods_only_connected_parts():
    passable = torch.zeros(1, 1, 9, 9)
    passable[0, 0, 1:4, 1:8] = 1     # region A
    passable[0, 0, 6:8, 1:8] = 1     # region B, separated by row 4-5
    seed = torch.zeros_like(passable)
    seed[0, 0, 2, 2] = 1
    r = ops.rec(seed, passable, k=20)
    assert r[0, 0, 1:4, 1:8].min() == 1 and r[0, 0, 6:8].sum() == 0


def test_smax_between_mean_and_max():
    x = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    region = torch.ones_like(x, dtype=torch.bool)
    s = float(ops.smax(x, region))
    assert 0.5 < s <= 1.0
    assert float(ops.smax(x, torch.zeros_like(region))) == 0.0


# ------------------------------------------------------------------ G1 with the REAL constants

from src.stage2.constraints.constraints import load_constants  # noqa: E402

REAL = load_constants()
REAL_CASES = toys.real_constant_cases(REAL)


def real_pair(name):
    """(valid, broken, hole) for a check under the real constants."""
    if name in REAL_CASES:
        v, b = REAL_CASES[name]
        return v, b, None
    b, hole = toys.broken(name)
    return toys.valid_plan(), b, hole


def real_ctx(name, valid, hole):
    if name in NO_GT:
        return Ctx(const=REAL, K=200)
    h = None if hole is None else torch.from_numpy(hole).float()[None, None]
    return Ctx(hole=h, Y=P_of(valid), const=REAL, K=200)


@pytest.mark.parametrize("name", NAMES)
def test_real_constants_valid_clean_broken_violates(name):
    valid, broken, hole = real_pair(name)
    ctx = real_ctx(name, valid, hole)
    assert float(violation_map(name, P_of(valid), ctx).max()) < 1e-4, f"{name} fires on the valid plan"
    assert float(violation_map(name, P_of(broken), ctx).sum()) > 0.5, f"{name} misses its broken plan"


@pytest.mark.parametrize("rule,fn", [("R1", r1_room_area), ("R3", r3_adjacency), ("R4", r4_exterior_door)])
def test_rules_real_constants(rule, fn):
    valid, broken, _ = real_pair(rule)
    assert not fn(valid, REAL)[0]
    assert fn(broken, REAL)[0]

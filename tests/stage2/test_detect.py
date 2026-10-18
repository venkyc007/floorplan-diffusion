"""Omission simulator and rule-based detector, on the hand-made toy plan (no dataset needed)."""

import numpy as np
import pytest

from src.stage2.constraints import toys
from src.stage2.data.classes16 import C16
from src.stage2.detect import rules
from src.stage2.detect.omissions import KINDS, MISSING, omit, parser_noise, simulate, simulate_one

I = C16.index


@pytest.mark.parametrize("kind", KINDS)
def test_each_omission_removes_its_component(kind):
    lab = toys.valid_plan()
    rng = np.random.default_rng(0)
    plan, mask, t = omit(lab, kind, rng)
    assert mask.any() and (plan[mask] != lab[mask]).any()
    assert (plan[~mask] == lab[~mask]).all()            # nothing outside the omitted region changes
    if kind == "window":
        assert (plan[mask] == I["wall"]).all()
    if kind == "fixture":
        assert MISSING[t] in C16.cfg["groups"]["F"] and np.isin(plan[mask], C16.group("S")).all()


def test_simulate_one_metadata():
    s = simulate_one(toys.valid_plan(), np.random.default_rng(1), "window")
    assert s["type"] == "window" and s["location"] == "exterior" and s["size_bin"] in ("small", "medium", "large")
    s = simulate_one(toys.valid_plan(), np.random.default_rng(1), "fixture")
    assert s["location"] == "interior"


def test_intact_share_and_noise_is_unlabelled():
    rng = np.random.default_rng(0)
    intact = sum(not simulate(toys.valid_plan(), rng)[1].any() for _ in range(200))
    assert 20 < intact < 70                                  # ~20%
    noisy = parser_noise(toys.valid_plan(), np.random.default_rng(0))
    assert 0 < (noisy != toys.valid_plan()).mean() < 0.05


def test_rules_quiet_on_valid_plan():
    cands = rules.detect(toys.valid_plan(), {r: 0.0 for r in rules.ALL_RULES})
    assert not [c for c in cands if set(c.rules) & {"C8", "C17", "R4"}]


def test_room_without_door_is_found_on_shared_wall():
    lab = toys.broken("C8")[0]                               # bathroom door walled up
    cands = rules.detect(lab, {r: 0.0 for r in rules.ALL_RULES})
    c8 = [c for c in cands if "C8" in c.rules]
    assert c8 and c8[0].type == "door"
    y0, x0, y1, x1 = c8[0].box
    assert y0 <= 48 and y1 >= 46                             # on the wall between bathroom and living room


def test_missing_toilet_and_bath_found():
    lab = toys.broken("C17")[0]
    cands = rules.detect(lab, {r: 0.0 for r in rules.ALL_RULES})
    assert any("C17" in c.rules and c.type == "toilet" for c in cands)


def test_no_exterior_door_found():
    cands = rules.detect(toys.broken("R4")[0], {r: 0.0 for r in rules.ALL_RULES})
    assert any("R4" in c.rules for c in cands)


def test_merge_overlapping():
    a = rules.Candidate("door", (0, 0, 10, 10), ["C8"], 5)
    b = rules.Candidate("wall", (5, 5, 20, 20), ["C1"], 2)
    c = rules.Candidate("window", (50, 50, 60, 60), ["C12"], 1)
    m = rules.merge([a, b, c])
    assert len(m) == 2 and m[0].type == "door" and m[0].box == (0, 0, 20, 20) and m[0].rules == ["C1", "C8"]

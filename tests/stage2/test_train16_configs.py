"""K1 ablation fairness: the constrained and control configs differ ONLY in constraints.enabled."""

import copy

import yaml

from src.utils import PROJECT_ROOT


def load(name):
    with open(PROJECT_ROOT / "configs" / "stage2" / f"train16_{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_only_the_switch_differs():
    a, b = load("constrained"), load("control")
    assert a["model"]["constraints"]["enabled"] is True
    assert b["model"]["constraints"]["enabled"] is False
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    a2["model"]["constraints"]["enabled"] = b2["model"]["constraints"]["enabled"] = None
    assert a2 == b2

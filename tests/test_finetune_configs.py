"""The constraint fine-tune and its control must be identical except for the
constraints switch, so the ablation is fair. Run:  python -m pytest tests/"""

import copy

import yaml

from src.utils import PROJECT_ROOT


def load(name):
    with open(PROJECT_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_control_differs_only_in_constraints_switch():
    on, off = load("finetune_constraints"), load("finetune_control")
    assert on["model"]["constraints"]["enabled"] is True
    assert off["model"]["constraints"]["enabled"] is False
    a, b = copy.deepcopy(on), copy.deepcopy(off)
    del a["model"]["constraints"]["enabled"], b["model"]["constraints"]["enabled"]
    assert a == b


def test_constraint_switches_parse_from_yaml():
    """Each loss has a `use` switch and a weight that survive YAML parsing
    (YAML reads a key named `on` as the boolean True)."""
    c = load("finetune_constraints")["model"]["constraints"]
    for name in ("cldice", "openings", "closure", "alignment"):
        assert c[name]["use"] is True and isinstance(c[name]["weight"], float), name


def test_finetune_settings():
    c = load("finetune_constraints")
    assert c["train"]["lr"] == 2.0e-5
    assert c["model"]["ddim_eta"] == 1.0
    base = load("train_diffusion")
    for key in ("block_out_channels", "layers_per_block", "attention_from_level", "num_train_timesteps"):
        assert c["model"][key] == base["model"][key], key   # same architecture as run 1 (needed for --init)

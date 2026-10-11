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
        assert isinstance(c[name]["use"], bool) and isinstance(c[name]["weight"], float), name
    # Fallback after the overfit test: only the ground-truth-based clDice + closure.
    assert [n for n in ("cldice", "openings", "closure", "alignment") if c[n]["use"]] == ["cldice", "closure"]


def test_finetune_settings():
    c = load("finetune_constraints")
    assert c["train"]["lr"] == 2.0e-5
    assert c["model"]["ddim_eta"] == 1.0
    base = load("train_diffusion")
    for key in ("block_out_channels", "layers_per_block", "attention_from_level", "num_train_timesteps"):
        assert c["model"][key] == base["model"][key], key   # same architecture as run 1 (needed for --init)


def test_matched_unet_differs_only_in_length():
    """The matched-sample U-Net must equal the original baseline except max_steps."""
    a, b = load("train_unet"), load("train_unet_matched")
    assert b["train"]["max_steps"] == 30000
    a["train"].pop("max_steps"); b["train"].pop("max_steps")
    assert a == b

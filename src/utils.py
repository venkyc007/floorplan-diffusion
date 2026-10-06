"""Small helpers shared by the whole project."""

from pathlib import Path

import yaml

# The project's top folder (the one containing src/, configs/, data/ ...).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(name):
    """Read configs/<name>.yaml and return it as a Python dictionary."""
    with open(PROJECT_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def project_path(relative):
    """Turn a path from a config (e.g. 'data/processed') into a full path."""
    return PROJECT_ROOT / relative

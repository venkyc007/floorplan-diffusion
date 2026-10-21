"""Unit tests for src/stage1/data/corruptions.py. Run:  python -m pytest tests/"""

import numpy as np

from src.stage1.data.corruptions import (TYPES, UNKNOWN, bbox, corrupt, padded_box_mask,
                                  to_model_input)
from tests.stage1.test_metrics import toy_plan

CFG = {
    "mask": {"padding": [2, 6]},
    "wall": {"run_length": 6, "segment_length": [8, 40]},
    "room": {"min_area": 20, "max_box_pct": 50},
    "region": {"side": [8, 16]},
    "position": {"edge_cutoff": 0.55},
    "size_bins": {"edges": [2.7, 7.7]},
}


def test_mask_is_padded_bounding_box():
    """The mask is the part's bounding box + 2..6 px on each side, never its shape."""
    part = np.zeros((128, 128), bool)
    part[50:52, 40:60] = True                      # a 2 x 20 wall piece
    r0, c0, r1, c1 = bbox(part)
    for seed in range(50):
        m = padded_box_mask(part, np.random.default_rng(seed), (2, 6), part.shape)
        mr0, mc0, mr1, mc1 = bbox(m)
        assert m.sum() == (mr1 - mr0) * (mc1 - mc0)            # a full rectangle
        assert 2 <= r0 - mr0 <= 6 and 2 <= c0 - mc0 <= 6
        assert 2 <= mr1 - r1 <= 6 and 2 <= mc1 - c1 <= 6


def test_every_type_on_toy_plan():
    plan = np.zeros((128, 128), np.uint8) + 5     # embed the 40x40 toy plan
    plan[40:80, 40:80] = toy_plan()
    for ctype in TYPES:
        mask, corrupted, meta = corrupt(plan, ctype, np.random.default_rng(0), CFG)
        assert meta["type"] == ctype
        assert ((corrupted == UNKNOWN) == mask).all()          # unknown exactly in the mask
        assert (corrupted[~mask] == plan[~mask]).all()         # rest untouched
        x = to_model_input(corrupted, mask)
        assert x.shape == (7, 128, 128)
        assert x[:6, mask].sum() == 0                          # no information in the mask
        assert (x[:6, ~mask].sum(axis=0) == 1).all()           # one-hot outside it
        assert (x[6] == mask).all()


def test_missing_type_returns_none():
    plan = np.zeros((128, 128), np.uint8) + 5
    plan[40:80, 40:80] = toy_plan()
    plan[plan == 2] = 0                                        # remove the window
    assert corrupt(plan, "window", np.random.default_rng(0), CFG) is None

"""Unit tests for src/stage1/stats.py (paired bootstrap). Run:  python -m pytest tests/"""

import numpy as np
import pandas as pd

from src.stage1.stats import bootstrap, verdict


def frame(tmiou, conn, rng):
    n = len(tmiou)
    return pd.DataFrame({"tmiou": tmiou, "conn_err": conn, "n_rooms_leaking": rng.integers(0, 2, n),
                         "n_rooms_in_mask": np.full(n, 2), "room_count_err": conn, "valid": rng.random(n) > 0.3})


def test_clear_difference_is_detected():
    rng = np.random.default_rng(0)
    base = rng.random(500)
    a, b = frame(base + 0.05, np.zeros(500), rng), frame(base, np.ones(500), rng)
    out = bootstrap(a, b, np.arange(500), np.random.default_rng(1))
    diff, lo, hi = out["tolerant mIoU"]
    assert abs(diff - 0.05) < 1e-9 and lo > 0
    assert verdict("tolerant mIoU", diff, lo, hi) == "clear: A better"
    diff, lo, hi = out["connectivity error"]
    assert hi < 0 and verdict("connectivity error", diff, lo, hi) == "clear: A better"   # lower is better


def test_noise_only_is_not_clear():
    """Per-case differences that are pure noise with exactly zero mean."""
    rng = np.random.default_rng(0)
    base, noise = rng.random(300), rng.normal(0, 0.05, 300)
    noise -= noise.mean()
    a, b = frame(base + noise, rng.random(300), rng), frame(base, rng.random(300), rng)
    out = bootstrap(a, b, np.arange(300), np.random.default_rng(1))
    diff, lo, hi = out["tolerant mIoU"]
    assert lo < 0 < hi

"""Omission metadata (src/stage2/data/omission_meta.py): location, position, size bin, and that every
image-level omission carries it."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.data.omission_meta import FIELDS, breakdown, omission_meta
from src.stage2.detect.image_eval import omit

I = C16.index


def plan():
    """100 x 100: background outside, a square building (rows / cols 10-89) with 3 px walls, one living room,
    a window on the left outer wall and a sink in the middle."""
    lbl = np.full((100, 100), I["background"], np.uint8)
    lbl[10:90, 10:90] = I["wall"]
    lbl[13:87, 13:87] = I["living"]
    lbl[40:56, 10:13] = I["window"]
    lbl[46:54, 46:54] = I["sink"]
    return lbl


def test_window_on_outer_wall_is_exterior_edge_small():
    lbl = plan()
    m = omission_meta(lbl, lbl == I["window"], "window")
    assert m["type"] == "window"
    assert m["location"] == "exterior"
    assert m["position"] == "edge"
    assert m["size_bin"] == "small"
    assert 0 < m["size_pct"] < 2.7


def test_fixture_in_the_middle_is_interior_centre():
    lbl = plan()
    m = omission_meta(lbl, lbl == I["sink"], "fixture")
    assert (m["location"], m["position"], m["size_bin"]) == ("interior", "centre", "small")
    assert m["center_dist"] < 0.55


def test_size_bins_follow_stage1_edges():
    lbl = plan()
    building = 80 * 80
    for side, want in ((10, "small"), (20, "medium"), (30, "large")):   # 1.6%, 6.3%, 14% of the building
        m = np.zeros(lbl.shape, bool)
        m[35:35 + side, 35:35 + side] = True
        meta = omission_meta(lbl, m, "room")
        assert meta["size_bin"] == want, (side, meta)
        assert abs(meta["size_pct"] - 100 * side * side / building) < 0.01


def test_intact_has_empty_fields():
    m = omission_meta(plan(), None, "intact")
    assert m["type"] == "intact" and m["location"] == "" and m["size_bin"] == ""


def test_every_image_level_omission_carries_metadata():
    lbl = plan()
    img = np.where(np.isin(lbl, [I["wall"]]), 0, 255).astype(np.uint8)
    o = omit("window", img, lbl, np.random.default_rng(0))
    assert o is not None and len(o) == 5
    meta = o[4]
    assert set(FIELDS) <= set(meta)
    assert meta["type"] == "window" and meta["location"] == "exterior"


def test_breakdown_splits_by_each_field():
    rows = [{"type": "window", "location": "exterior", "position": "edge", "size_bin": "small", "ok": 1},
            {"type": "fixture", "location": "interior", "position": "centre", "size_bin": "small", "ok": 0},
            {"type": "door (a)", "location": "interior", "position": "centre", "size_bin": "medium", "ok": 1}]
    cols = {"hits": lambda rs: str(sum(r["ok"] for r in rs))}
    t = breakdown(rows, "location", cols)
    assert t[2] == "| interior | 2 | 1 |" and t[3] == "| exterior | 1 | 1 |"
    t = breakdown(rows, "type", cols)
    assert [line.split(" | ")[0][2:] for line in t[2:]] == ["door (a)", "window", "fixture"]   # fixed order
    for f in FIELDS:
        assert len(breakdown(rows, f, cols)) >= 3

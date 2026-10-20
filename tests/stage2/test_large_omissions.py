"""Large omissions (room (whole), region) and the size-balanced mix for the test evaluation (user, 2026-10-20)."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.data.erasure_large import erase_region, erase_room
from src.stage2.detect.image_eval import omit
from src.stage2.detect.omission_mix import omit_in_bin, plan_bins

I = C16.index
ROOMS = ["bedroom", "bathroom", "kitchen", "other_room"]


def grid_plan(n=4, cell=60, wall=8, margin=20):
    """n x n rooms of `cell` px with `wall` px walls, a door in every interior wall; drawing: black walls on white."""
    size = n * cell + wall + 2 * margin
    lbl = np.full((size, size), I["background"], np.uint8)
    img = np.full((size, size), 255, np.uint8)
    b0, b1 = margin, margin + n * cell + wall
    lbl[b0:b1, b0:b1] = I["wall"]
    for r in range(n):
        for c in range(n):
            y, x = margin + wall + r * cell, margin + wall + c * cell
            lbl[y:y + cell - wall, x:x + cell - wall] = I[ROOMS[(r + c) % 4]]
            if c < n - 1:
                lbl[y + 20:y + 34, x + cell - wall:x + cell] = I["door"]
    img[lbl == I["wall"]] = 0
    img[lbl == I["door"]] = 255
    return img, lbl


def test_room_whole_keeps_outer_walls_and_merges():
    img, lbl = grid_plan()
    out, structure, area = erase_room(img, lbl, np.random.default_rng(0))
    outer = np.zeros(lbl.shape, bool)
    outer[20:28] = True                                     # top outer wall row band
    assert np.array_equal(out[outer & (lbl == I["wall"])], img[outer & (lbl == I["wall"])])
    walls = structure & (lbl == I["wall"])
    assert walls.any() and float(out[walls].mean()) > 200   # interior walls gone (paper)
    assert area.sum() > structure.sum()


def test_region_erases_inside_only():
    img, lbl = grid_plan()
    o = erase_region(img, lbl, np.random.default_rng(1))
    assert o is not None
    out, structure, area = o
    building = ~np.isin(lbl, C16.group("OUT"))
    pct = 100 * area.sum() / building.sum()
    assert 5 <= pct <= 25
    far = np.ones(lbl.shape, bool)
    ys, xs = np.nonzero(area)
    far[max(0, ys.min() - 60):ys.max() + 60, max(0, xs.min() - 60):xs.max() + 60] = False
    assert np.array_equal(out[far], img[far])


def test_omit_large_kinds_carry_metadata():
    img, lbl = grid_plan()
    for kind in ("room (whole)", "region"):
        o = omit(kind, img, lbl, np.random.default_rng(2))
        assert o is not None and o[2] == "room" and o[4]["type"] == kind
        assert o[4]["size_bin"] in ("medium", "large")


def test_size_balanced_mix():
    assert plan_bins(9) == ["small", "medium", "large"] * 3
    b = plan_bins(10)
    assert sorted(b.count(x) for x in ("small", "medium", "large")) == [3, 3, 4]
    img, lbl = grid_plan()
    rng = np.random.default_rng(3)
    for target in ("small", "medium", "large"):
        o = omit_in_bin(target, img, lbl, rng)
        assert o is not None and o[4]["size_bin"] == target

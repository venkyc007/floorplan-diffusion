"""Parse vs drawing double check (user, 2026-10-20): window drawn as plain wall / no ink, parsed wall with no ink,
the 40% style guard, and room holes never getting a door on an exterior wall."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.detect.symbol_ink import (wall_ink_candidates, wall_ink_runs, window_features,
                                          window_ink_candidates, window_verdict)
from src.stage2.pipeline.run import existing, restrict

I = C16.index
F = 8 / 3                                  # drawing px per label px (walls 8 px drawn, 3 px parsed)


def scene(window_style):
    """Label (60 x 60, walls 3 px) with a horizontal exterior wall at rows 30-32 and a window at x 24-35; the
    drawing at 8/3 scale: solid dark wall, the window drawn as thin lines / as wall / not at all."""
    lab = np.full((60, 60), I["bedroom"], np.uint8)
    lab[:30] = I["background"]
    lab[30:33] = I["wall"]
    lab[30:33, 24:36] = I["window"]
    H = int(60 * F)
    img = np.full((H, H), 255, np.uint8)
    r0, r1 = int(30 * F), int(33 * F)
    img[r0:r1] = 40                                                   # the wall line, solid
    c0, c1 = int(24 * F), int(36 * F)
    if window_style == "lines":
        img[r0:r1, c0:c1] = 255
        img[r0, c0:c1] = img[r1 - 1, c0:c1] = img[(r0 + r1) // 2, c0:c1] = 0
    elif window_style == "gap":
        img[r0:r1, c0:c1] = 255
    return lab, img


def test_window_features_separate_the_three_cases():
    v = {}
    for style in ("lines", "wall", "gap"):
        lab, img = scene(style)
        (d, f), = window_features(img, lab)
        v[style] = window_verdict(f, d_max=0.05, gap_ink=0.02)
        if style == "lines":
            assert f["runs"] >= 2
    assert v == {"lines": None, "wall": "wall", "gap": "gap"}


def test_window_candidate_names_its_component_and_guard():
    lab, img = scene("wall")
    c, = window_ink_candidates(img, lab, 0.05, 0.02, guard=1.0)
    assert c.type == "window" and c.rules == ["WINK"] and c.mask.sum() == 36
    assert window_ink_candidates(img, lab, 0.05, 0.02) == []        # 1 of 1 windows lacks its symbol: > 40%


def test_wall_without_ink_is_found():
    lab, img = scene("lines")
    img[int(30 * F):int(33 * F), int(5 * F):int(15 * F)] = 255       # the drawing has a gap the parse filled
    runs, share = wall_ink_runs(img, lab)
    assert max(n for _, n in runs) >= 8 and share < 0.4
    c = wall_ink_candidates(img, lab, min_len=6)
    assert len(c) == 1 and c[0].type == "wall" and c[0].mask[31, 10]
    h = {"type": "wall", "rules": ["WALLINK"], "box": c[0].box, "point": c[0].peak, "mask": c[0].mask}
    assert existing(lab, h) is not None                               # reported as "wall symbol missing", marked


def test_room_hole_never_gets_an_exterior_door():
    before = np.full((40, 40), I["bedroom"], np.uint8)
    before[:10] = I["background"]
    before[10:13] = I["wall"]
    before[25:28, 20:40] = I["wall"]
    before[28:] = I["kitchen"]
    after = before.copy()
    after[10:13, 5:15] = I["door"]                                    # exterior wall (background on one face)
    after[25:28, 25:35] = I["door"]                                   # between bedroom and kitchen
    kept = restrict(before, after, "room", point=(11, 10))           # the point is at the exterior door
    assert not (kept[10:13] == I["door"]).any()
    assert (kept[25:28, 25:35] == I["door"]).any()

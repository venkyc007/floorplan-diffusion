"""R7 (gap too narrow for a doorway) and R6's exterior-run measurement on small toy plans (completion scale)."""

import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.detect.narrow_gaps import gap_candidates_narrow
from src.stage2.detect.window_runs import length, runs

I = C16.index


def toy_wall():
    lab = np.full((40, 60), I["bedroom"], np.uint8)
    lab[:10] = I["hallway"]
    lab[10:13, :] = I["wall"]
    return lab


def test_narrow_gap_found_door_gap_not():
    lab = toy_wall()
    lab[10:13, 20:23] = I["bedroom"]                  # 3 px break: narrower than any door -> wall missing
    lab[10:13, 35:50] = I["door"]                      # a real door: no candidate
    c = gap_candidates_narrow(lab)
    assert len(c) == 1 and c[0].type == "wall" and c[0].box[1] <= 20 <= c[0].box[3]


def test_wide_gap_is_not_r7():
    lab = toy_wall()
    lab[10:13, 20:32] = I["bedroom"]                  # 12 px: a doorway-sized gap is R5's business
    assert gap_candidates_narrow(lab) == []


def test_exterior_run_length():
    lab = np.full((40, 60), I["background"], np.uint8)
    lab[5:35, 5:55] = I["wall"]
    lab[8:32, 8:52] = I["bedroom"]                    # walls 3 px, outside all around
    lab[5:8, 20:30] = I["window"]                     # a window in the top wall breaks the top run
    r = [(d, length(m, d)) for n, d, m in runs(lab, need_area=1514) if d == "up"]
    assert sorted(L for _, L in r) == [12, 22]

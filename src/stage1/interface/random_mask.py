"""Random mask for the Stage 1 tab: remove one part of a complete plan with the benchmark's own rules.

    plan, mask, truth, text = random_corruption(complete_plan, seed)

The corruption is src/stage1/data/corruptions.corrupt with configs/stage1/corruption.yaml, i.e. exactly how the training data
and the val / test benchmarks were made: one door, window, wall segment, room or random region removed, and the
mask = its bounding box + random 2-6 px padding per side (never the part's exact shape). The type is drawn at
random; if the plan has no part of that type the next type is tried.
"""

import numpy as np
import yaml

from src.stage1.data.corruptions import TYPES, UNKNOWN, corrupt
from src.utils import project_path

CFG = yaml.safe_load(open(project_path("configs/stage1/corruption.yaml"), encoding="utf-8"))


def random_corruption(complete, seed, types=TYPES):
    """(corrupted plan with 255 = unknown, mask, true plan, description), or None if no type fits the plan."""
    if (complete == UNKNOWN).any():
        raise ValueError("the plan still has unknown (grey) pixels; a random mask needs a complete plan")
    rng = np.random.default_rng(int(seed))
    start = int(rng.integers(len(types)))
    for ctype in list(types[start:]) + list(types[:start]):
        out = corrupt(complete, ctype, rng, CFG)
        if out is not None:
            mask, corrupted, meta = out
            size = meta.get("size_bin", "")
            return corrupted, mask, complete.copy(), f"random {ctype} removed ({size} hole, seed {seed})"
    return None

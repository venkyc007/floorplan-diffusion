"""'Suggest suspicious areas': propose repair masks around architectural rule violations.

ASSISTIVE ONLY, NOT EXHAUSTIVE: it runs four of our architectural rules
(src/stage1/eval/arch_metrics.py) on the whole input plan and puts a padded box
around every violation it finds:
  unreachable rooms, dead-end walls, floating wall fragments, doors that do
  not connect two spaces.
Real plans also break these rules sometimes (e.g. 11% of real val plans have a
dead-end wall, often next to an open passage), and many kinds of damage break
none of them, so the user should review and edit the proposed mask.
"""

import numpy as np

from src.stage1.data.classes import BACKGROUND
from src.stage1.data.corruptions import UNKNOWN
from src.stage1.eval.arch_metrics import violation_locations

SUGGEST_RULES = {
    "unreachable": "unreachable room",
    "dead_ends": "dead-end wall",
    "fragments": "floating wall fragment",
    "door_spaces": "door not between two spaces",
}


def padded_box(spot, pad, shape):
    rows, cols = np.nonzero(spot)
    box = np.zeros(shape, bool)
    box[max(0, rows.min() - pad):rows.max() + pad + 1, max(0, cols.min() - pad):cols.max() + pad + 1] = True
    return box


def suggest(plan, thresholds, pad=4):
    """(mask, findings): union of padded boxes around violations, and a list of
    (rule name, box centre (row, col)). Already-unknown pixels are always included."""
    unknown = plan == UNKNOWN
    label = np.where(unknown, BACKGROUND, plan).astype(np.uint8)  # rules need a complete map
    found = violation_locations(label, np.ones(plan.shape, bool), thresholds)
    mask, findings = unknown.copy(), []
    for rule, name in SUGGEST_RULES.items():
        for spot in found[rule]:
            if (spot & unknown).any():          # inside the unknown area: an artefact of filling it in
                continue
            mask |= padded_box(spot, pad, plan.shape)
            r, c = np.argwhere(spot).mean(0)
            findings.append((name, (int(r), int(c))))
    return mask, findings

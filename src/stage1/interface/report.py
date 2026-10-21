"""Metrics shown after a repair.

Always: the 6 architectural validity rules (src/stage1/eval/arch_metrics.py),
counted inside the repaired region (the mask grown by 3 px, as in the
evaluation). Without a ground truth, a rule passes if it has 0 violations
there. With a ground truth (gallery plans), a rule passes if the repair has
no more violations than the true plan in the same region — the definition of
architectural validity used in the evaluation.

With a ground truth also: tolerant mIoU and door/window/room component F1
inside the mask (src/stage1/eval/metrics.py).
"""

import numpy as np

from src.stage1.data.classes import CLASSES
from src.stage1.eval.arch_metrics import MARGIN, RULES, grow, label_metrics
from src.stage1.eval.metrics import DETECT_CLASSES, sample_stats

RULE_NAMES = {
    "dead_ends": "no dead-end walls",
    "fragments": "no floating wall fragments",
    "opening_bridge": "doors/windows bridge two wall ends",
    "door_spaces": "doors connect two spaces",
    "window_out": "windows face outside",
    "unreachable": "every room reachable through doors",
}


def rule_table(repaired, mask, thresholds, gt=None):
    """[(rule name, repaired count, ground-truth count or None, passed)] for the 6 rules."""
    region = grow(mask, MARGIN)
    pred = label_metrics(repaired, region, thresholds)
    true = label_metrics(gt, region, thresholds) if gt is not None else None
    rows = []
    for rule in RULES:
        g = true[rule] if true is not None else None
        rows.append((RULE_NAMES[rule], int(pred[rule]), g, pred[rule] <= (g if g is not None else 0)))
    return rows


def f1(tp, fp, fn):
    return None if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def gt_scores(repaired, gt, mask):
    """Tolerant mIoU + component F1 per class inside the mask (None = class absent in both)."""
    s = sample_stats(repaired, gt, mask)
    out = {"tolerant mIoU": None if np.isnan(s["tmiou"]) else s["tmiou"]}
    for c in DETECT_CLASSES:
        name = CLASSES[c]
        out[f"{name} F1"] = f1(s[f"tp_{name}"], s[f"fp_{name}"], s[f"fn_{name}"])
    return out


def to_markdown(rows, scores=None, runtime=None, how=None):
    has_gt = rows and rows[0][2] is not None
    lines = []
    if how:
        lines += [f"**{how}**", ""]
    if runtime is not None:
        lines += [f"Runtime: **{runtime:.2f} s**", ""]
    n_pass = sum(r[3] for r in rows)
    lines += [f"### Architectural rules in the repaired region: {n_pass}/6 pass", ""]
    if has_gt:
        lines += ["| rule | repair | true plan | |", "|---|---|---|---|"]
        lines += [f"| {n} | {p} | {g} | {'✅ pass' if ok else '❌ fail'} |" for n, p, g, ok in rows]
        lines += ["", "_A rule passes if the repair has no more violations than the true plan in the same "
                      "region (real plans break some rules too)._"]
    else:
        lines += ["| rule | violations | |", "|---|---|---|"]
        lines += [f"| {n} | {p} | {'✅ pass' if ok else '❌ fail'} |" for n, p, _, ok in rows]
        lines += ["", "_No true plan is known, so a rule passes only with 0 violations; real plans break "
                      "some rules too (e.g. 11% have a dead-end wall next to an open passage)._"]
    if scores:
        lines += ["", "### Compared with the true plan (inside the mask)", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {'n/a (none in the hole)' if v is None else f'{v:.3f}'} |" for k, v in scores.items()]
    return "\n".join(lines)

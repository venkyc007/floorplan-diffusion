"""Paired bootstrap confidence intervals for method differences (Day 6, C). CPU only.

Reads the per-case results written by src/stage1/arch_eval.py
(outputs/stage1/arch_eval/per_case/<setting>.csv.gz; same val cases, same order for
every setting) and, for each comparison (A - B), resamples the CASES with
replacement 1,000 times. In every resample both methods are scored on the
SAME cases, exactly as the reports define each metric:
  tolerant mIoU, connectivity error, room count error  -> mean over cases
  room leak                                           -> pooled: leaking rooms / rooms
  architectural validity                              -> % of valid cases
The 2.5th-97.5th percentiles of the differences give a 95% confidence
interval; a difference is "statistically clear" if the interval excludes 0.

    python -m src.stage1.stats            -> outputs/stage1/stats.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import project_path

N_RESAMPLES = 1000
SEED = 0
# (label, metric function on per-case arrays, "higher is better")
METRICS = {
    "tolerant mIoU": (lambda d, i: np.nanmean(d["tmiou"][i]), True),
    "connectivity error": (lambda d, i: d["conn_err"][i].mean(), False),
    "room leak": (lambda d, i: d["n_rooms_leaking"][i].sum() / max(d["n_rooms_in_mask"][i].sum(), 1), False),
    "room count error": (lambda d, i: d["room_count_err"][i].mean(), False),
    "architectural validity %": (lambda d, i: 100 * d["valid"][i].mean(), True),
}
COMPARISONS = [
    ("constraints - control", "ft_constraints_refine600", "ft_control_refine600"),
    ("best diffusion - U-Net", "ft_constraints_refine600", "unet"),
    ("best diffusion - matched U-Net", "ft_constraints_refine600", "unet_matched"),
]
SUBSETS = [("overall", None), ("type = room", ("type", "room")), ("size = large", ("size_bin", "large"))]


def load(per_case_dir, name):
    path = Path(per_case_dir) / f"{name}.csv.gz"
    return pd.read_csv(path) if path.exists() else None


def arrays(df):
    cols = ["tmiou", "conn_err", "n_rooms_leaking", "n_rooms_in_mask", "room_count_err", "valid"]
    return {c: df[c].to_numpy(dtype=float) for c in cols}


def bootstrap(a, b, rows, rng):
    """{metric: (difference, low, high)} for A - B on the given case rows."""
    da, db = arrays(a), arrays(b)
    samples = rng.integers(0, len(rows), size=(N_RESAMPLES, len(rows)))
    out = {}
    for name, (fn, _) in METRICS.items():
        diff = fn(da, rows) - fn(db, rows)
        boots = np.array([fn(da, rows[s]) - fn(db, rows[s]) for s in samples])
        out[name] = (diff, *np.percentile(boots, [2.5, 97.5]))
    return out


def verdict(name, diff, lo, hi):
    if lo <= 0 <= hi:
        return "not clear (interval includes 0)"
    better = (diff > 0) == METRICS[name][1]
    return "clear: A better" if better else "clear: A worse"


def plain_summary(label, name_a, results):
    """One sentence per subset: which metrics are clearly better / worse / unclear for A."""
    lines = [f"- **{label}** (A = {name_a}):"]
    for sub_label, res in results:
        groups = {"clear: A better": [], "clear: A worse": [], "not clear (interval includes 0)": []}
        for metric, (diff, lo, hi) in res.items():
            groups[verdict(metric, diff, lo, hi)].append(metric)
        parts = [f"better on {', '.join(groups['clear: A better']) or 'nothing'}",
                 f"worse on {', '.join(groups['clear: A worse']) or 'nothing'}"]
        if groups["not clear (interval includes 0)"]:
            parts.append(f"no clear difference on {', '.join(groups['not clear (interval includes 0)'])}")
        lines.append(f"  - {sub_label}: A is " + "; ".join(parts) + ".")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-case", default="outputs/stage1/arch_eval/per_case")
    parser.add_argument("--out", default="outputs/stage1/stats.md")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")      # the Windows console defaults to cp1252
    rng = np.random.default_rng(SEED)
    lines = ["# Statistics: paired bootstrap over val cases", "",
             f"{N_RESAMPLES} resamples of the cases (with replacement, seed {SEED}); both methods are scored on the "
             "same resampled cases each time. Columns: observed difference A − B and its 95% interval. "
             "A difference is **statistically clear** only if the interval excludes 0. "
             "For connectivity error, room leak and room count error, LOWER is better.", ""]
    summary, details = ["## Summary: which differences are statistically clear", ""], []
    for label, name_a, name_b in COMPARISONS:
        a, b = load(project_path(args.per_case), name_a), load(project_path(args.per_case), name_b)
        if a is None or b is None:
            details += [f"## {label}", "", f"_Skipped: per-case results for {name_a if a is None else name_b} "
                      "not available yet._", ""]
            continue
        assert (a["plan_id"].to_numpy() == b["plan_id"].to_numpy()).all(), "cases are not paired"
        details += [f"## {label}  (A = {name_a}, B = {name_b})", "",
                    "| subset | metric | A − B | 95% interval | verdict |", "|---|---|---|---|---|"]
        results = []
        for sub_label, sub in SUBSETS:
            rows = np.arange(len(a)) if sub is None else np.flatnonzero(a[sub[0]].to_numpy() == sub[1])
            res = bootstrap(a, b, rows, rng)
            results.append((sub_label, res))
            for metric, (diff, lo, hi) in res.items():
                details.append(f"| {sub_label} (n={len(rows)}) | {metric} | {diff:+.4f} | "
                               f"[{lo:+.4f}, {hi:+.4f}] | {verdict(metric, diff, lo, hi)} |")
        details.append("")
        summary += plain_summary(label, name_a, results)
    lines += summary + [""] + details
    project_path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

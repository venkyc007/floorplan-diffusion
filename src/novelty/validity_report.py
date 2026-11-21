"""Novelty round 3 - validity table: how rule-following are the ADDITIONS of each pipeline version?

    python -m src.novelty.validity_report outputs/novelty/e2e_tuning_v3d_*.csv --out outputs/novelty/validity_tuning.md

Reads the per-case rows written by src/novelty/eval_e2e.py (which now stores the counts of
src/novelty/regularise.checks for every medium / large case, for the plan and for the true layout) and prints one
table per omission size: accuracy next to validity, one line per arm, plus a line for the true layout (real
drawings do not score zero on these counts, so the truth is the reference, not zero).
How it fits in: a reader of CSVs only; it runs no model.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

CHECKS = ["added_wall_px", "ragged", "thick_sd", "loose", "specks", "patches"]
HEAD = ("| arm | n | recovered (2 px) | fixed (strict) | rule violations (truth) | valid | added wall px | not straight | "
        "thickness spread, px | loose wall groups | specks | small room patches |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [r for p in args.csv for r in csv.DictReader(open(p, encoding="utf-8"))]
    T = lambda v: str(v) == "True"
    mean = lambda R, f: float(np.mean([float(r[f]) for r in R if r.get(f, "") != ""] or [np.nan]))
    L = ["# Validity of what is added (medium and large omissions)", "",
         "Counts on the walls / pieces a plan adds inside the omission's box (src/novelty/regularise.checks); "
         "rule violations and `valid` are the project's own rule check (src/stage2/pipeline/e2e_val.py: `valid` = "
         "not more violations than the true layout).", ""]
    for target in ("medium", "large"):
        sub = [r for r in rows if r["target"] == target and r.get("v_added_wall_px", "") != ""]
        if not sub:
            continue
        L += [f"## {target} omissions", "", HEAD, "|" + "---|" * 12]
        arms = list(dict.fromkeys(r["arm"] for r in sub))
        for arm in arms:
            R = [r for r in sub if r["arm"] == arm]
            L.append(f"| {arm} | {len(R)} | {100 * np.mean([T(r['recovered']) for r in R]):.0f}% | "
                     f"{100 * np.mean([T(r['fixed']) for r in R]):.0f}% | {mean(R, 'viol_after'):.1f} ({mean(R, 'viol_gt'):.1f}) | "
                     f"{100 * np.mean([T(r['valid_after']) for r in R]):.0f}% | {mean(R, 'v_added_wall_px'):.0f} | "
                     f"{100 * mean(R, 'v_ragged'):.1f}% | {mean(R, 'v_thick_sd'):.2f} | {mean(R, 'v_loose'):.2f} | "
                     f"{mean(R, 'v_specks'):.2f} | {mean(R, 'v_patches'):.2f} |")
        R = [r for r in sub if r["arm"] == arms[0]]
        L.append(f"| true layout | {len(R)} | - | - | {mean(R, 'viol_gt'):.1f} | 100% | {mean(R, 't_added_wall_px'):.0f} | "
                 f"{100 * mean(R, 't_ragged'):.1f}% | {mean(R, 't_thick_sd'):.2f} | {mean(R, 't_loose'):.2f} | "
                 f"{mean(R, 't_specks'):.2f} | {mean(R, 't_patches'):.2f} |")
        L.append("")
    text = "\n".join(L)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

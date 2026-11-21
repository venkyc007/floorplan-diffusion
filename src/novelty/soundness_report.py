"""Accuracy next to structural soundness for every pipeline version (round 4 tables).

    python -m src.novelty.soundness_report <e2e csv files> --out outputs/novelty/soundness_tuning.md [--names a=b ...]

Reads the rows written by src/novelty/eval_e2e.py (one arm per version; rename arms with --names "old=new") and
writes, for medium and large omissions (a file given as "file.csv::name" has all its rows named "name"):
  accuracy   found, recovered (2 px), fixed (strict), tolerant mIoU in the region
  soundness  the three checks of src/novelty/soundness.py (walls connect, openings in walls, rooms closed and
             reachable) and "sound" = all three; plus the count of room-type meetings without a wall (mixed)
  rule check the project's own rule check (src/stage2/pipeline/e2e_val.rule_violations), summed in four groups,
             and "valid vs oracle" = no more violations than the ORACLE (the parser's map with the true layout
             pasted into the erased area: what a perfect fill would score; the clean true drawing scores far fewer
             because it has no parser errors around the hole)
Reference lines (same drawings): the parser's map given (nothing filled), the oracle and the true layout.
Paired differences between two arms (bootstrap 95% intervals) with --pair "arm A" "arm B".
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = {"walls (C1-C5)": ["C1", "C2", "C3", "C4", "C5"],
          "openings (C9-C13)": ["C9", "C10", "C11", "C12", "C13"],
          "rooms (C6-C8, R1, R3)": ["C6", "C7", "C8", "R1", "R3"],
          "fixtures / railings (C14-C18)": ["C14", "C15", "C16", "C17", "C18"]}
CHECKS = [("walls connect", "walls_connect"), ("openings in walls", "openings_in_walls"),
          ("rooms closed", "rooms_closed"), ("sound", "sound")]


def T(v):
    return str(v) == "True"


def mean(R, f, conv=float):
    v = [conv(r[f]) for r in R if r.get(f, "") != ""]
    return float(np.mean(v)) if v else float("nan")


def pct(R, f):
    v = [T(r[f]) for r in R if r.get(f, "") != ""]
    return f"{100 * np.mean(v):.0f}%" if v else "-"


def groups_of(js):
    d = json.loads(js) if js else {}
    return {g: sum(d.get(k, 0) for k in ks) for g, ks in GROUPS.items()}


def paired(rows, a, b, target, f, scale=100, conv=T, seed=0):
    A = {r["pid"]: conv(r[f]) for r in rows if r["arm"] == a and r["target"] == target and r.get(f, "") != ""}
    B = {r["pid"]: conv(r[f]) for r in rows if r["arm"] == b and r["target"] == target and r.get(f, "") != ""}
    ids = sorted(set(A) & set(B))
    if not ids:
        return "-"
    d = np.array([float(B[i]) - float(A[i]) for i in ids]) * scale
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return f"{d.mean():+.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", nargs="*", default=[], help='"old arm=new arm"')
    ap.add_argument("--pair", nargs=2, action="append", default=[])
    ap.add_argument("--title", default="Accuracy and structural soundness")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    ren = dict(n.split("=", 1) for n in a.names)
    rows = []
    for p in a.csvs:
        p, _, label = p.partition("::")               # "file.csv::v4" sets the arm name of every row of that file
        for r in csv.DictReader(open(p, encoding="utf-8")):
            r["arm"] = label or ren.get(r["arm"], r["arm"])
            rows.append(r)
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    L = [f"# {a.title}", ""]
    if a.note:
        L += [a.note, ""]
    L += ["Soundness = the three structural checks of the review on the omission's region (src/novelty/soundness.py), "
          "judged without the true layout. Rule check = the project's 18-rule check (violation blobs in a window "
          "around the omission), in four groups; 'valid vs oracle' = no more violations than the parser's map with "
          "the true layout pasted into the erased area.", ""]
    for target in ("large", "medium"):
        L += [f"## {target} omissions", ""]
        head = ("| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | " +
                " | ".join(t for t, _ in CHECKS) + " | mixed rooms | rule violations | " +
                " | ".join(GROUPS) + " | valid vs oracle |")
        L += [head, "|" + "---|" * (head.count("|") - 1)]
        ref = None
        for arm in arms:
            R = [r for r in rows if r["arm"] == arm and r["target"] == target]
            if not R:
                continue
            ref = ref or R
            g = defaultdict(float)
            for r in R:
                for k, v in groups_of(r.get("rules_after", "")).items():
                    g[k] += v / len(R)
            vo = [int(r["viol_after"]) <= int(r["viol_oracle"]) for r in R if r.get("viol_oracle", "") != ""]
            L.append(f"| {arm} | {len(R)} | {pct(R, 'found')} | {pct(R, 'recovered')} | {pct(R, 'fixed')} | "
                     f"{mean(R, 'tol_after'):.1f} | " + " | ".join(pct(R, 's_' + k) for _, k in CHECKS) +
                     f" | {mean(R, 's_mixed'):.2f} | {mean(R, 'viol_after'):.1f} | " +
                     " | ".join(f"{g[k]:.1f}" for k in GROUPS) +
                     (f" | {100 * np.mean(vo):.0f}% |" if vo else " | - |"))
        if ref:
            for name, p, viol, rules in (("parser's map given (nothing filled)", "sp_", None, None),
                                         ("oracle (true layout pasted into the erased area)", "so_", "viol_oracle", "rules_oracle"),
                                         ("true layout", "st_", "viol_gt", None)):
                g = defaultdict(float)
                if rules:
                    for r in ref:
                        for k, v in groups_of(r.get(rules, "")).items():
                            g[k] += v / len(ref)
                L.append(f"| *{name}* | {len(ref)} | - | - | - | - | " +
                         " | ".join(pct(ref, p + k) for _, k in CHECKS) + f" | {mean(ref, p + 'mixed'):.2f} | " +
                         (f"{mean(ref, viol):.1f}" if viol else "-") + " | " +
                         " | ".join(f"{g[k]:.1f}" if rules else "-" for k in GROUPS) + " | - |")
        L += ["", "Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that "
              "does not match is a different but structurally valid layout (the review's 'plausible alternative').", "",
              "| version | sound and matching the truth | sound, different layout | not sound |", "|---|---|---|---|"]
        for arm in arms:
            R = [r for r in rows if r["arm"] == arm and r["target"] == target and r.get("s_sound", "") != ""]
            if R:
                sm = np.mean([T(r["s_sound"]) and T(r["recovered"]) for r in R])
                sa = np.mean([T(r["s_sound"]) and not T(r["recovered"]) for r in R])
                L.append(f"| {arm} | {100 * sm:.0f}% | {100 * sa:.0f}% | {100 * (1 - sm - sa):.0f}% |")
        L.append("")
        for x, y in a.pair:
            L += [f"Paired, {y} minus {x} ({target}): recovered {paired(rows, x, y, target, 'recovered')}, fixed "
                  f"{paired(rows, x, y, target, 'fixed')}, sound {paired(rows, x, y, target, 's_sound')}, rule violations "
                  f"{paired(rows, x, y, target, 'viol_after', 1, float)}.", ""]
    it = [r for r in rows if r["target"] == "intact"]
    if it:
        L += ["## intact drawings (nothing erased)", "", "| version | n | drawings with a region flag | added components / drawing |",
              "|---|---|---|---|"]
        for arm in arms:
            R = [r for r in it if r["arm"] == arm]
            if R:
                L.append(f"| {arm} | {len(R)} | {100 * np.mean([int(r['n_region_flags']) > 0 for r in R]):.1f}% | "
                         f"{mean(R, 'added'):.2f} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Completion rate broken down by omission size, with the definition of a "completed" omission (review request).

    python -m src.novelty.completion_by_size --out outputs/novelty/completion_by_size.md

Why: the review asked to break the reported 46% completion rate down by omission size and to define what counts as a
completed omission. The 46% comes from the second Stage 2 TEST evaluation (one single-component omission per test
drawing, railings left out; outputs/stage2/test_components.md). Almost all of those omissions are "small" by the
project's size bins (< 2.7% of the building's area), so this script splits them finer by area, and adds the
size-balanced Stage 2 test (small / medium / large, outputs/stage2/final_test.md) and the novelty results on the
held-out validation drawings (large omissions).

It only RE-READS saved per-case results (outputs/kaggle/test_components, outputs/kaggle/final_test,
outputs/novelty/*.csv); it runs no model and does not touch the test drawings again.
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from src.utils import PROJECT_ROOT

COMP = PROJECT_ROOT / "outputs" / "kaggle" / "test_components" / "test_components"
FINAL = PROJECT_ROOT / "outputs" / "kaggle" / "final_test" / "final_test"
NOV = PROJECT_ROOT / "outputs" / "novelty"

DEFINITIONS = """## What counts as a completed omission

One omission is erased from a real drawing; the system gets only the damaged drawing (no mask, no hint).

- **found**: the CHECK stage puts a flag (a marked box) on the erased spot.
- **completed ("fixed", the 46% figure)**: found, AND the final map has the erased component's class back at the
  erased spot with IoU >= 0.3 between the erased pixels and the plan's pixels of that class there (whoever drew
  them: the fill or the parser). For a wall-type omission (wall piece, a room's walls, whole room, region) the class
  is wall.
- **recovered (2 px)** (large omissions, novelty work): found, AND at least half of the erased wall pixels have a
  wall of the plan within 2 px, AND at least half of the walls the plan adds there lie within 2 px of a true wall.
  Used because walls are 3 px thick, so a wall drawn one or two pixels off scores almost no IoU; the strict rule
  above tops out at 72% even when the true layout is pasted back (parser maps are not pixel-exact).
- **sound** (round 4, see `soundness_*.md`): the completed region passes the three structural checks (walls
  connect, openings sit in walls, rooms closed and reachable). Judged without the true layout, so a plausible
  alternative to the true layout can pass.

Sizes: omitted area as a share of the building's area: small < 2.7%, medium 2.7-7.7%, large >= 7.7% (the
project's bins, src/stage2/data/omission_meta.py). Brackets: Wilson 95% intervals.
"""


def wilson(k, n, z=1.96):
    if n == 0:
        return "-"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.0f}% [{max(0.0, 100 * (c - h)):.0f}, {min(100.0, 100 * (c + h)):.0f}]"


def jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(str(pattern))):
        rows += [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
    return rows


def T(v):
    return v is True or str(v) == "True"


def table(groups, cols):
    """groups: [(name, rows)], cols: [(title, key)] -> markdown lines."""
    out = ["| omissions | n | " + " | ".join(t for t, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, R in groups:
        cells = []
        for _, k in cols:
            S = [r for r in R if r.get(k, "") != ""]       # "recovered" exists only for wall-type omissions
            cells.append(wilson(sum(T(r[k]) for r in S), len(S)) + ("" if len(S) == len(R) else f" (n {len(S)})"))
        out.append(f"| {name} | {len(R)} | " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(NOV / "completion_by_size.md"))
    a = ap.parse_args()
    L = ["# Completion rate by omission size", "", DEFINITIONS]

    # 1. the 46%: component-level test
    C = [r for r in jsonl(COMP / "part*.jsonl") if r.get("kind") != "railing"]
    pct = lambda r: float(r["meta_size_pct"] or 0)
    groups = [("all except railings (the 46%)", C),
              ("small, < 0.25% of the building", [r for r in C if pct(r) < 0.25]),
              ("small, 0.25-1%", [r for r in C if 0.25 <= pct(r) < 1]),
              ("small, 1-2.7%", [r for r in C if 1 <= pct(r) < 2.7]),
              ("medium, 2.7-7.7%", [r for r in C if 2.7 <= pct(r) < 7.7]),
              ("large, >= 7.7%", [r for r in C if pct(r) >= 7.7])]
    L += ["## 1. The 46%: Stage 2 TEST, one single-component omission per drawing (2026-10-22)", "",
          "330 test drawings (railings left out: the system handles them only in manual mode). Single components "
          "are almost always small, so the small bin is split by area.", ""]
    L += table(groups, [("found", "found"), ("completed (fixed)", "fixed"), ("drawn by the fill", "fixed_add")])
    kinds = sorted({r["kind"] for r in C})
    L += ["", "By kind and size (completed / n):", "", "| kind | < 0.25% | 0.25-1% | 1-2.7% | >= 2.7% |", "|---|---|---|---|---|"]
    for k in kinds:
        R = [r for r in C if r["kind"] == k]
        cells = []
        for lo, hi in ((0, 0.25), (0.25, 1), (1, 2.7), (2.7, 1e9)):
            S = [r for r in R if lo <= pct(r) < hi]
            cells.append(f"{sum(T(r['fixed']) for r in S)}/{len(S)}" if S else "-")
        L.append(f"| {k} | " + " | ".join(cells) + " |")

    # 2. size-balanced end-to-end test
    E = [r for r in jsonl(FINAL / "part*.jsonl") if r.get("section") == "e2e"]
    L += ["", "## 2. Stage 2 TEST, size-balanced end to end (2026-10-22, shipped pipeline)", "",
          "120 test drawings, a third each small / medium / large omissions of mixed kinds (doors, windows, walls, "
          "fixtures, railings, whole rooms, erased regions).", ""]
    L += table([("all", E)] + [(b, [r for r in E if r["target_bin"] == b]) for b in ("small", "medium", "large")],
               [("found", "found"), ("completed (fixed)", "fixed"), ("drawn by the fill", "fixed_add")])

    # 3. novelty, held-out val
    L += ["", "## 3. Large omissions after the novelty work (held-out VALIDATION drawings, 151, test not used)", "",
          "Same definitions; one small, one medium and one large omission per drawing (separate runs).", ""]
    R = list(csv.DictReader(open(NOV / "e2e_heldout_all_arms.csv", encoding="utf-8")))
    for name, arm in (("shipped pipeline", "shipped auto"), ("round 1 hybrid (v2)", "hybrid v2 auto"),
                      ("final hybrid v3c (round 2)", "hybrid v3c auto")):
        A = [r for r in R if r["arm"] == arm]
        L.append(f"**{name}**:")
        L.append("")
        L += table([(t, [r for r in A if r["target"] == t]) for t in ("small", "medium", "large")],
                   [("found", "found"), ("completed (fixed, strict)", "fixed"), ("recovered (2 px)", "recovered")])
        L.append("")
    R4 = NOV / "e2e_heldout_round4.csv"                   # round 4 (structurally sound fills), kernel v16
    if R4.exists():
        A = [r for r in csv.DictReader(open(R4, encoding="utf-8")) if r["arm"] == "hybrid v4c auto"]
        L += ["**round 4 final v4c (structurally sound fills)**:", ""]
        L += table([(t, [r for r in A if r["target"] == t]) for t in ("small", "medium", "large")],
                   [("found", "found"), ("completed (fixed, strict)", "fixed"), ("recovered (2 px)", "recovered"),
                    ("sound (3 structural checks)", "s_sound")])
        L.append("")
    L += ["Reading: completion falls with size in the shipped system (single small components about 2 in 5, large "
          "erased regions almost never). The novelty work targets the large end; small omissions are unchanged by it. "
          "Round 4 trades some accuracy for structurally sound fills (`outputs/novelty/soundness_heldout.md`)."]
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

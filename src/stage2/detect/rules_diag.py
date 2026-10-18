"""Rule diagnostics (user, 2026-10-18), appended to outputs/stage2/rules_check.md.

    python -m src.stage2.detect.rules_diag --zip completion_val.zip [--part ceiling|ta|split|all]

ceiling  For the 287 single omissions of `rules_eval check` (same seeds), an omission is
         "rule-visible" if ANY rule's raw evidence (before thresholds) inside the omitted region
         (grown 3 px) is higher on the omitted plan than on the intact plan: summed violation for
         the map rules, number of overlapping raw candidates for the discrete rules (C8, C17, R1, R3, R4).
         Reported per type: % rule-visible, and the thresholded detector's recall among those.
ta       The TA's case: 100 val plans where a room's ONLY door is replaced by wall. C8 recall =
         share where a raw C8 candidate (and a thresholded one) overlaps that room or the door.
split    Re-split the false-alarm budget: C8, C17, C6, C12 get 0.1 per plan each, the other rules
         share the remaining 0.1 (greedy as in `tune`); writes configs/stage2/detect_rules.yaml and
         re-runs the omission check.
"""

import argparse

import cv2
import numpy as np
import yaml

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.detect import rules as R
from src.stage2.detect.omissions import KINDS, simulate_one
from src.stage2.detect.rules_eval import OUT, check, ordered_plans

I = C16.index


def grown(m, px=3):
    return cv2.dilate(m.astype(np.uint8), np.ones((2 * px + 1,) * 2, np.uint8)) > 0


def evidence(label, region):
    """{rule: evidence inside region} (map rules: summed V; discrete: overlapping raw candidates)."""
    maps = R.violation_maps(label)
    ev = {r: float(v[region].sum()) for r, v in maps.items()}
    for c in R.discrete_candidates(label):
        y0, x0, y1, x1 = c.box
        if region[y0:y1 + 1, x0:x1 + 1].any():
            ev[c.rules[0]] = ev.get(c.rules[0], 0) + 1
    return ev


def omissions(zip_path, n=300):
    """The same omissions as rules_eval.check (seed 1, kinds in turn)."""
    plans = ordered_plans(zip_path)
    rng = np.random.default_rng(1)
    for i in range(n):
        pid, lab = plans[rng.integers(len(plans))]
        s = simulate_one(lab, rng, KINDS[i % len(KINDS)])
        if s is not None:
            yield lab, s


def ceiling(zip_path, th):
    rows = []
    for lab, s in omissions(zip_path):
        region = grown(s["mask"])
        e0, e1 = evidence(lab, region), evidence(s["plan"], region)
        visible = [r for r in e1 if e1[r] > e0.get(r, 0) + 1e-6]
        cands = R.detect(s["plan"], th)
        found = any(s["mask"][c.box[0]:c.box[2] + 1, c.box[1]:c.box[3] + 1].any() for c in cands)
        rows.append((s["kind"], bool(visible), found, visible))
    L = ["", "## Diagnostic 1: rule-visibility ceiling", "",
         "Rule-visible = some rule's raw evidence (before thresholds) inside the omitted region (+3 px) is higher "
         "after the omission than before. Recall among visible = thresholded detector finds it.", "",
         "| kind | n | rule-visible | recall among visible | most frequent visible rules |", "|---|---|---|---|---|"]
    for k in KINDS:
        sel = [r for r in rows if r[0] == k]
        vis = [r for r in sel if r[1]]
        freq = {}
        for r in vis:
            for rule in r[3]:
                freq[rule] = freq.get(rule, 0) + 1
        top = ", ".join(f"{a} {b}" for a, b in sorted(freq.items(), key=lambda x: -x[1])[:4])
        L.append(f"| {k} | {len(sel)} | {100 * len(vis) / max(len(sel), 1):.0f}% | "
                 f"{100 * sum(r[2] for r in vis) / max(len(vis), 1):.0f}% | {top} |")
    allv = [r for r in rows if r[1]]
    L.append(f"| all | {len(rows)} | {100 * len(allv) / len(rows):.0f}% | {100 * sum(r[2] for r in allv) / max(len(allv), 1):.0f}% | |")
    return L


def single_door_rooms(label):
    """[(room mask, its only door mask)] for room instances touched by exactly one door component."""
    k = 3
    out = []
    doors = components(label == I["door"])
    for c, m in R.room_instances(label):
        ring = grown(m, k)
        touching = [d for d in doors if (d & ring).any()]
        if len(touching) == 1:
            out.append((m, touching[0]))
    return out


def ta_case(zip_path, th, n=100):
    plans = ordered_plans(zip_path)
    rng = np.random.default_rng(3)
    raw_hit = th_hit = tried = 0
    for pid, lab in plans:
        if tried >= n:
            break
        cases = single_door_rooms(lab)
        if not cases:
            continue
        room, door = cases[rng.integers(len(cases))]
        plan = lab.copy()
        plan[door] = I["wall"]
        target = grown(room | door, 3)
        raw = [c for c in R.discrete_candidates(plan) if c.rules[0] == "C8"]
        thr = [c for c in R.detect(plan, th) if "C8" in c.rules]
        hit = lambda cs: any(target[c.box[0]:c.box[2] + 1, c.box[1]:c.box[3] + 1].any() for c in cs)
        raw_hit += hit(raw)
        th_hit += hit(thr)
        tried += 1
    return tried, raw_hit / max(tried, 1), th_hit / max(tried, 1)


def split_budget(zip_path, priority=("C8", "C17", "C6", "C12"), each=0.1, total=0.5):
    plans = ordered_plans(zip_path)[:200]
    scores = {r: [] for r in R.ALL_RULES}
    for _, lab in plans:
        for c in R.raw_candidates(lab):
            scores[c.rules[0]].append(c.score)
    n = len(plans)

    def threshold_for(srt, cap):
        """Smallest threshold keeping <= cap * n candidates."""
        keep = int(np.floor(cap * n))
        if len(srt) <= keep:
            return 0.0, len(srt)
        th = srt[len(srt) - keep] if keep > 0 else float("inf")
        kept = sum(s >= th for s in srt)
        while kept > keep:                     # ties: move above them
            above = [s for s in srt if s > th]
            th = above[0] if above else float("inf")
            kept = len(above)
        return th, kept
    srt = {r: sorted(v) for r, v in scores.items()}
    th, kept = {}, {}
    for r in priority:
        th[r], kept[r] = threshold_for(srt[r], each)
    rest = [r for r in R.ALL_RULES if r not in priority]
    budget = total - sum(kept[r] for r in priority) / n
    for r in rest:
        th[r], kept[r] = 0.0, len(srt[r])
    while sum(kept[r] for r in rest) / n > budget:
        r = max(rest, key=lambda x: kept[x])
        lo = srt[r][len(srt[r]) - kept[r]]
        above = [s for s in srt[r] if s > lo]
        th[r] = above[0] if above else float("inf")
        kept[r] = len(above)
    doc = {"thresholds": {r: (float(v) if np.isfinite(v) else 1e12) for r, v in th.items()},
           "false_alarms_per_plan": {r: round(kept[r] / n, 3) for r in R.ALL_RULES},
           "raw_candidates_per_plan": {r: round(len(srt[r]) / n, 3) for r in R.ALL_RULES},
           "source": f"{n} intact typed val plans; budget split (user 2026-10-18): {', '.join(priority)} {each} "
                     f"per plan each, the other rules share the rest of {total}"}
    R.THRESHOLDS_FILE.write_text("# GENERATED by python -m src.stage2.detect.rules_diag --part split\n" +
                                 yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--part", default="all", choices=["ceiling", "ta", "split", "all"])
    args = ap.parse_args()
    md = OUT / "rules_check.md"
    L = []
    if args.part == "all":          # thresholds first, then the main check table with them, then the diagnostics
        doc = split_budget(args.zip)
        check(argparse.Namespace(zip=args.zip, n=300))
        args.part = "all_after_split"
        L += ["", "## Diagnostic 3: false-alarm budget re-split (C8, C17, C6, C12: 0.1 per plan each)", "",
              "The tables at the top use these thresholds. Before the re-split (equal ~0.03 per rule, and C8 starting "
              "from the largest walkable area) the rules detected 7% (type-correct 3%) at 0.38 false alarms per plan.", "",
              "| rule | threshold | false alarms / plan (tuning) |", "|---|---|---|"]
        L += [f"| {r} | {doc['thresholds'][r]:.4g} | {doc['false_alarms_per_plan'][r]} |" for r in R.ALL_RULES]
    if args.part in ("ta", "all", "all_after_split"):
        tried, raw, thr = ta_case(args.zip, R.load_thresholds())
        L += ["", "## Diagnostic 2: the TA case (a room's ONLY door replaced by wall)", "",
              f"{tried} val plans. C8 recall: **raw {100 * raw:.0f}%**, with the thresholds in force {100 * thr:.0f}%."]
        print(L[-1], flush=True)
    if args.part in ("ceiling", "all", "all_after_split"):
        L += ceiling(args.zip, R.load_thresholds())
        print("\n".join(L[-10:]), flush=True)
    if args.part == "split":
        doc = split_budget(args.zip)
        L += ["", "## Diagnostic 3: false-alarm budget re-split (C8, C17, C6, C12: 0.1 per plan each)", "",
              "New thresholds / false alarms per plan on the 200 tuning plans: " +
              ", ".join(f"{r} {doc['thresholds'][r]:.4g} / {doc['false_alarms_per_plan'][r]}" for r in R.ALL_RULES) + ".",
              "", "The omission check with these thresholds is the main table at the top of this file (re-run)."]
    with open(md, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

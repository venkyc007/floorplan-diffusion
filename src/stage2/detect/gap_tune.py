"""Choose the final R5 (doorless gap) cut-offs on the TUNING plans only (label maps; no models, light).

    python -m src.stage2.detect.gap_tune --zip completion_val.zip
        -> outputs/stage2/gap_rule.md, configs/stage2/detector_threshold.yaml (gap_rule, extra_rules += R5)

Tuning plans = the plans that chose the detector setting (detection_eval.build_cases: 100 intact + 200
single omissions). Room-size cut-offs (user): need_area = 25th percentile of bedroom instance areas,
small_other = 25th percentile of other_room instance areas, both over the tuning plans' rooms.
Budget: the total CHECK (detector 0.9 / 40 px + C8 + R5) must stay <= 0.4 false alarms per intact plan.
Detector + C8 is at 0.31 (held-out, already reported), so R5 may add at most 0.09 per intact plan, measured
here on the tuning intact plans (R5 counted on its own, so overlap with detector boxes is not credited:
conservative). If the rule alone exceeds that, the shortest firing gap length is raised (Lmin_door upward)
until it fits. Recall is shown for door kind (b) (doorway left open) on the tuning omissions.
"""

import argparse

import numpy as np
import yaml

from src.stage2.constraints.rules import room_instances
from src.stage2.data.classes16 import C16
from src.stage2.detect.detection_eval import CHOICE, OUT, build_cases, hit, write_choice
from src.stage2.detect.gaps import gap_candidates
from src.stage2.detect.rules import CONST, merge

BUDGET, CHECK_FA = 0.4, 0.31


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    cases = build_cases(args.zip)
    areas = {"bedroom": [], "other_room": []}
    seen = set()
    for c in cases:                                   # every tuning plan's rooms, each plan once
        if c["pid"] in seen:
            continue
        seen.add(c["pid"])
        for cls, m in room_instances(c.get("orig", c["plan"])):
            if C16.names[cls] in areas:
                areas[C16.names[cls]].append(int(m.sum()))
    need, small = int(np.percentile(areas["bedroom"], 25)), int(np.percentile(areas["other_room"], 25))
    intact = [c for c in cases if c["mask"] is None]
    door_b = [c for c in cases if c["kind"] == "door" and not (c["plan"][c["mask"]] == C16.index["wall"]).all()]
    rows, chosen = [], None
    for min_len in range(CONST["Lmin_door"], CONST["Lmax_door"] + 1):
        kw = {"need_area": need, "small_other": small, "min_len": min_len}
        fa = np.mean([len(merge(gap_candidates(c["plan"], **kw))) for c in intact])
        rec = np.mean([hit(merge(gap_candidates(c["plan"], **kw)), c["mask"], "door")[0] for c in door_b])
        pair_only = min_len == CONST["Lmin_door"] and np.mean([len(merge(gap_candidates(c["plan"]))) for c in intact])
        rows.append((min_len, fa, rec, pair_only))
        if fa <= BUDGET - CHECK_FA:
            chosen = (min_len, fa, rec)
            break
    L = ["# Doorless-gap rule R5: final cut-offs (tuning plans only)", "",
         f"Tuning plans: {len(seen)} (100 intact, {len(door_b)} door kind (b) omissions). Room-size cut-offs: "
         f"need_area = p25 of {len(areas['bedroom'])} bedroom areas = {need} px; small_other = p25 of "
         f"{len(areas['other_room'])} other_room areas = {small} px (completion scale, walls 3 px).",
         f"Pair rule alone (rate < 10%, no size condition): {rows[0][3]:.2f} false alarms per intact plan.",
         f"Budget for R5: {BUDGET} - {CHECK_FA} (detector + C8, held-out) = {BUDGET - CHECK_FA:.2f} per intact plan.", "",
         "| shortest gap (px) | R5 false alarms / intact plan | door (b) recall |", "|---|---|---|"]
    L += [f"| {r[0]} | {r[1]:.2f} | {100 * r[2]:.0f}% |" for r in rows]
    if chosen:
        L += ["", f"Chosen: shortest gap {chosen[0]} px -> {chosen[1]:.2f} false alarms, door (b) recall {100 * chosen[2]:.0f}% "
              "(tuning plans)."]
        choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
        choice["extra_rules"] = sorted(set(choice.get("extra_rules", [])) | {"R5"})
        choice["gap_rule"] = {"open_rate": 0.10, "need_area": need, "small_other": small, "min_len": chosen[0],
                              "source": "gap_tune.py on the tuning plans; R5 alone <= 0.09 false alarms per intact plan"}
        write_choice(choice)
    else:
        L += ["", "No cut-off fits the budget: R5 stays OFF."]
    (OUT / "gap_rule.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

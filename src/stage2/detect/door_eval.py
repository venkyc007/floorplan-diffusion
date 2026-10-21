"""CHECK with the two kinds of door omission reported separately (user, chunk 8).

    python -m src.stage2.detect.door_eval --zip completion_val.zip --ckpt <k2a_detector best.pt>
        -> outputs/stage2/door_eval.md, configs/stage2/detector_threshold.yaml (gap_rule, extra_rules)

Door omissions: (a) "opening forgotten" = wall across the doorway (caught by C8 when the room is cut off);
(b) "gap left, no door" = the doorway stays open (caught by the doorless-gap rule R5, detect/gaps.py).
Methods: detector (0.9 / 40 px) + C8 (the current CHECK) and detector + C8 + R5.
R5 variants, loosest first: pairs with contact rate < 10% (the user's rule), then without other_room
(closets / storage rooms are often doorless), then rate < 5%, then < 2%. Chosen on the TUNING split
(build_cases: 100 intact + 200 single omissions): the loosest variant whose total false alarms per intact
plan stay <= the budget (0.4); reported on the 159 held-out plans (build_heldout_cases).
"""

import argparse
from collections import defaultdict

import numpy as np
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.detect import rules as R
from src.stage2.detect.detection_eval import (CHOICE, OUT, write_choice, build_cases, build_heldout_cases, detector_probs, hit,
                                              load_detector, proposals)
from src.stage2.detect.gaps import gap_candidates

KINDS = ["door (a)", "door (b)", "window", "wall", "railing", "fixture", "room"]
VARIANTS = [("rate < 10%", 0.10, ()), ("rate < 10%, no other_room", 0.10, ("other_room",)),
            ("rate < 5%, no other_room", 0.05, ("other_room",)), ("rate < 2%, no other_room", 0.02, ("other_room",))]


def kind_of(c):
    if c["mask"] is None:
        return None
    if c["kind"] == "door":
        return "door (a)" if (c["plan"][c["mask"]] == C16.index["wall"]).all() else "door (b)"
    return c["kind"]


def prepare(cases, model, K, th, mb):
    """Cache per case: detector proposals, C8 proposals, R5 candidates per variant (no probabilities kept)."""
    th_rules = R.load_thresholds()
    for i, c in enumerate(cases):
        c["k"] = kind_of(c)
        c["det"] = proposals(detector_probs(model, c["plan"], K), th, mb)
        c["c8"] = R.detect(c["plan"], th_rules, raw=[r for r in R.discrete_candidates(c["plan"]) if r.rules[0] == "C8"])
        c["r5"] = {v[0]: gap_candidates(c["plan"], open_rate=v[1], skip=v[2]) for v in VARIANTS}
        if (i + 1) % 50 == 0:
            print(f"prepared {i + 1}/{len(cases)}", flush=True)


def score(cases, variant=None):
    """-> (false alarms per intact plan, {kind: (found, type-correct)}, mean recall)."""
    fa, n_int = 0, 0
    per = defaultdict(lambda: [0, 0, 0])
    for c in cases:
        props = R.merge(c["det"] + c["c8"] + (c["r5"][variant] if variant else []))
        if c["k"] is None:
            fa += len(props)
            n_int += 1
        else:
            f, t = hit(props, c["mask"], c["type"])
            per[c["k"]][0] += 1
            per[c["k"]][1] += f
            per[c["k"]][2] += t
    rec = {k: (v[1] / v[0], v[2] / v[0], v[0]) for k, v in per.items()}
    return fa / max(n_int, 1), rec, float(np.mean([r[0] for r in rec.values()]))


def row(name, res):
    fa, rec, mean = res
    cells = [f"{100 * rec[k][0]:.0f}% ({100 * rec[k][1]:.0f}%)" if k in rec else "-" for k in KINDS]
    return f"| {name} | {fa:.2f} | {100 * mean:.0f}% | " + " | ".join(cells) + " |"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget", type=float, default=0.4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    model, cfg = load_detector(args.ckpt)
    choice = yaml.safe_load(CHOICE.read_text(encoding="utf-8"))
    th, mb = choice["threshold"], choice["min_blob"]
    tune = build_cases(args.zip)
    held, _ = build_heldout_cases(args.zip)
    prepare(tune + held, model, cfg["model"]["K"], th, mb)

    tune_res = {None: score(tune)} | {v[0]: score(tune, v[0]) for v in VARIANTS}
    ok = [v[0] for v in VARIANTS if tune_res[v[0]][0] <= args.budget]
    chosen = ok[0] if ok else None
    hdr = ["| method | false alarms / intact plan | mean recall | " + " | ".join(KINDS) + " |",
           "|---|---|---|" + "---|" * len(KINDS)]
    n_t = {k: sum(c["k"] == k for c in tune) for k in KINDS}
    n_h = {k: sum(c["k"] == k for c in held) for k in KINDS}
    L = [f"# Door omissions (a) / (b) and the doorless-gap rule R5{args.tag}", "",
         "(a) opening forgotten = wall across the doorway (C8); (b) gap left, no door = doorway left open (R5, "
         "src/stage2/detect/gaps.py). Detector 0.9 / 40 px. Recall = a proposal box overlaps the omission "
         "(in brackets: with the right type).", "",
         f"## Tuning split (choice; 100 intact plans; cases per kind: " + ", ".join(f"{k} {n_t[k]}" for k in KINDS) + ")", ""]
    L += hdr + [row("detector + C8", tune_res[None])] + [row(f"detector + C8 + R5 ({v[0]})", tune_res[v[0]]) for v in VARIANTS]
    L += ["", f"Chosen: **{chosen or 'none (no R5 variant fits the budget)'}** (loosest R5 variant with <= {args.budget} "
          "false alarms per intact plan on the tuning split).", "",
          f"## Held-out (159 plans; 59 intact; cases per kind: " + ", ".join(f"{k} {n_h[k]}" for k in KINDS) + ")", ""]
    L += hdr + [row("detector + C8", score(held))]
    if chosen:
        L.append(row(f"detector + C8 + R5 ({chosen})", score(held, chosen)))
    (OUT / "door_eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if chosen:
        v = next(v for v in VARIANTS if v[0] == chosen)
        choice["extra_rules"] = sorted(set(choice.get("extra_rules", [])) | {"R5"})
        choice["gap_rule"] = {"open_rate": v[1], "skip": list(v[2]), "source": "door_eval.py, tuning split, budget "
                              f"{args.budget} false alarms per intact plan"}
        write_choice(choice)
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""Gate G2: how often REAL plans break each constraint (spec section 4).

    python -m src.stage2.complete.gate_g2 --zip completion_train.zip [--n 1000] [--seed 0]
        -> outputs/stage2/constraint_gt_violations.md (+ .csv)

For 1,000 ground-truth training windows (128 x 128, completion scale, around random typed
deletions) every constraint's violation map V(Y) is computed on the ground truth itself, in
its ABSOLUTE (inference) form, inside M = hole grown 3 px. A window "violates" if the summed
violation in M is >= 1 (about one fully violating pixel).

Rule (spec): an absolute-mode constraint violated in > 10% of windows switches to GT-hinged;
> 40% means it is not a rule in this data -> proposed to DROP (the user decides).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.windows import model_inputs
from src.stage2.constraints import ops
from src.stage2.constraints.constraints import REGISTRY, Ctx, violation_map
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2" / "constraint_gt_violations"


def window_violations(labels, masks, batch=50):
    """{constraint: (per-window summed V in M, per-window mean V in M)} on the ground truth."""
    res = {n: ([], []) for n in REGISTRY}
    for i in range(0, len(labels), batch):
        cond, Y, mask = model_inputs(labels[i:i + batch], masks[i:i + batch])
        M = ops.dil(mask, 7)
        for name in REGISTRY:
            if name == "C17":                 # GT-relative by construction: 0 on the ground truth
                v = torch.zeros_like(mask)
            elif name in ("C1", "C8"):        # their inference (absolute) form, no ground truth
                v = violation_map(name, Y, Ctx(hole=mask if name != "C8" else None, K=96))
            else:
                v = violation_map(name, Y, Ctx(hole=mask, Y=None, known=Y * (1 - mask), K=96))
            res[name][0].append((v * M).flatten(1).sum(1).numpy())
            res[name][1].append(ops.masked_mean(v, M).numpy())
    return {n: (np.concatenate(a), np.concatenate(b)) for n, (a, b) in res.items()}


def decide(mode, share):
    if mode != "absolute":
        return f"keep {mode}"
    if share > 0.40:
        return "PROPOSE DROP (ask the user)"
    if share > 0.10:
        return "switch to gt_hinged"
    return "keep absolute"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    plans = load_plans(args.zip)
    labels, masks, kinds, _ = fixed_windows(plans, args.n, args.seed)
    viol = window_violations(labels, masks)
    rows = []
    for name, spec in REGISTRY.items():
        s, m = viol[name]
        share = float((s >= 1).mean())
        rows.append({"constraint": name, "title": spec.title, "default mode": spec.mode,
                     "% windows violating": round(100 * share, 1), "mean V in M": f"{m.mean():.5f}",
                     "decision": decide(spec.mode, share)})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(OUT) + ".csv", index=False, encoding="utf-8")
    by_type = pd.Series(kinds).value_counts().to_dict()
    lines = ["# Gate G2: constraint violations on real plans", "",
             f"{args.n} ground-truth training windows (128 x 128, completion scale; seed {args.seed}) around random "
             f"typed deletions ({', '.join(f'{k} {v}' for k, v in by_type.items())}), from {len(plans)} plans. "
             "V computed on the ground truth in its absolute (inference) form inside M = hole grown 3 px; "
             "a window violates if the summed V in M is >= 1. C17 is GT-relative (0 on the ground truth). "
             "Rule: absolute mode violated in > 10% of windows -> GT-hinged; > 40% -> propose to drop (user decides).",
             "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    Path(str(OUT) + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

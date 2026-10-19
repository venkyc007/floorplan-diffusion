"""Val plan ids: TUNING (used to choose detector / rule settings, and parser v2 checkpoints) vs HELD-OUT.

    python -m src.stage2.parse.val_split --zip completion_val.zip --out <dir>   -> tuning_ids.txt, heldout_ids.txt

Same definition as src/stage2/detect/detection_eval.py: tuning = the plans build_cases draws from;
held-out = every other typed val plan (build_heldout_cases uses these, 159 plans).
"""

import argparse
from pathlib import Path

from src.stage2.detect.detection_eval import build_cases
from src.stage2.detect.rules_eval import ordered_plans


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    used = {c["pid"] for c in build_cases(args.zip)}
    plans = [p for p, _ in ordered_plans(args.zip)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuning_ids.txt").write_text("\n".join(p for p in plans if p in used) + "\n", encoding="utf-8")
    (out / "heldout_ids.txt").write_text("\n".join(p for p in plans if p not in used) + "\n", encoding="utf-8")
    print(f"tuning {len(used)} plans, held-out {len(plans) - len(used)} plans")


if __name__ == "__main__":
    main()

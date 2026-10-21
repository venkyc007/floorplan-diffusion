"""Kaggle kernel floorplan-stage2-judge (CPU, no GPU): generous flagger + U-Net judge (src/stage2/detect/judge.py).

  1. tune on parsed TUNING drawings -> configs/stage2/judge.yaml (+ outputs/stage2/judge_tune.md)
  2. held-out drawings: current final CHECK (ink tier 2 off) vs generous + judge -> judge_eval.md, judge_scores.png

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data, -parser-v2, -k2a, -k1,
-detv2, -detv3 (checkpoints).
Output: /kaggle/working/stage2/ (judge.yaml, judge_tune.md, judge_eval.md, judge_scores.png).
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"


def copy_code():
    hits = [h for h in glob.glob("/kaggle/input/**/src/utils.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import run
    from src.stage2.kaggle_cpu import finish, setup
    data = setup()
    try:
        run(f"{sys.executable} -m src.stage2.detect.judge tune --zip {data}/parser_val.zip "
            f"--tuning configs/stage2/val_split/tuning_ids.txt")
        shutil.copy(f"{CODE}/configs/stage2/judge.yaml", f"{CODE}/outputs/stage2/judge.yaml")
        run(f"{sys.executable} -m src.stage2.detect.judge eval --completion {data}/completion_val.zip "
            f"--drawings {data}/parser_val.zip")
    finally:
        finish()

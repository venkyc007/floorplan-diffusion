"""Kaggle kernel floorplan-stage2-eval-tune (CPU, no GPU): re-tune the whole CHECK on parsed tuning drawings
(src/stage2/detect/check_tune.py, 40 intact + 80 omitted tuning drawings, parser v2B)

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips),
floorplan-stage2-parser-v1, floorplan-stage2-k2a, floorplan-stage2-k1 (checkpoints).
Output: /kaggle/working/stage2/ (check_tune.md); the chosen configs/stage2/detector_threshold.yaml is copied there too.
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
        run(f"{sys.executable} -m src.stage2.detect.check_tune --zip {data}/parser_val.zip --write "
            f"--tuning configs/stage2/val_split/tuning_ids.txt")
        shutil.copy(f"{CODE}/configs/stage2/detector_threshold.yaml", f"{CODE}/outputs/stage2/detector_threshold.yaml")
    finally:
        finish()

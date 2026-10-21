"""Kaggle kernel floorplan-stage2-eval-c8 (CPU, no GPU): why C8 misses cut-off rooms on parsed drawings
(src/stage2/detect/c8_diag.py, 15 held-out drawings + colorful-1261)

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips),
floorplan-stage2-parser-v1, floorplan-stage2-k2a, floorplan-stage2-k1 (checkpoints).
Output: /kaggle/working/stage2/ (c8_diag.md, c8_diag.png).
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
        run(f"{sys.executable} -m src.stage2.detect.c8_diag --zip {data}/parser_val.zip "
            f"--heldout configs/stage2/val_split/heldout_ids.txt --n 15 --extra colorful-1261")
    finally:
        finish()

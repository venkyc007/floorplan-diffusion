"""Kaggle kernel floorplan-stage2-eval-e2e (CPU): end-to-end validation runs (src/stage2/pipeline/e2e_val.py).
Version history: v1-v3 batch 1 (+ check set), v4 batch 1 + check set + batch 2; now: batch 3 only (4 room (whole),
4 region, 4 small omissions on 12 fresh held-out drawings -> e2e_val_batch3).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips) and the
checkpoint kernels. Output: /kaggle/working/stage2/ (e2e_val_batch3.png / .md / .csv).
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
        run(f"{sys.executable} -m src.stage2.pipeline.e2e_val --zip {data}/parser_val.zip --batch 3 --tag _batch3")
    finally:
        finish()

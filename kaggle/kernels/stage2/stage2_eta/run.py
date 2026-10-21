"""Kaggle kernel floorplan-stage2-eta (GPU T4): guidance strength eta on 150 val holes (tuning plans), fast hybrid.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips) and the
checkpoint kernels (setup() copies them where configs/stage2/pipeline.yaml expects them).
Output: /kaggle/working/stage2/ (eta_tune.md, eta_tune.json).
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
        run(f"{sys.executable} -m src.stage2.complete.eta_tune --zip {data}/completion_val.zip")
    finally:
        finish()

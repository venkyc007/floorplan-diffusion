"""Kaggle kernel floorplan-stage2-scale-diag (CPU): what the scale mismatch costs the FILL, with the hole given
(src/stage2/complete/scale_diag.py: 30 small + 30 large val holes, each filled at the training scale and at the
pipeline's 0.838x scale; fast settings).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of floorplan-stage2-data (completion_val.zip) and of the
Stage 2 checkpoint kernels (K2a U-Net, K1 diffusion, B4; the parser and detectors are loaded by run.models() but not
used here). Output: /kaggle/working/stage2/scale_diag.md / .csv.
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
        run(f"{sys.executable} -m src.stage2.complete.scale_diag --data {data}")
    finally:
        finish()

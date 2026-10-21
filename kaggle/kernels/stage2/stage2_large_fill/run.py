"""Kaggle kernel floorplan-stage2-large-fill (CPU): the three large-hole fills compared with the hole GIVEN (manual
mode): src/stage2/pipeline/large_fill_compare.py --mode manual (tag _manual).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips), the Stage 2
checkpoint kernels and the frozen Stage 1 kernels floorplan-unet-matched / floorplan-finetune-constraints (B2).
Output: /kaggle/working/stage2/ (large_fill_compare_manual.md / .csv / .png).
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"
MODE = "manual"


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
        run(f"{sys.executable} -m src.stage2.pipeline.large_fill_compare --zip {data}/parser_val.zip --mode {MODE} "
            f"--tag _{MODE}")
    finally:
        finish()

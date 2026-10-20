"""Kaggle kernel floorplan-stage2-eval-e2e (CPU): end-to-end validation on 12 new held-out val drawings, then the
12-drawing check set (tag _v4: + window / wall ink double checks), then batch 2 = 12 fresh held-out drawings (e2e_val_batch2).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (val zips) and the
checkpoint kernels. Output: /kaggle/working/stage2/ (e2e_val.png, e2e_val.md, e2e_val.csv).
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
        run(f"{sys.executable} -m src.stage2.pipeline.e2e_val --zip {data}/parser_val.zip --tag _v4")
        run(f"{sys.executable} -m src.stage2.pipeline.check_set --zip {data}/parser_val.zip --tag _v4")
        run(f"{sys.executable} -m src.stage2.pipeline.e2e_val --zip {data}/parser_val.zip --batch 2 --tag _batch2")
    finally:
        finish()

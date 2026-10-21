"""Kaggle kernel floorplan-stage2-b4-val (CPU): B4 validation with the hole given (src/stage2/complete/b4_val.py), then
the 20-drawing large-hole comparison with the B4 models (manual + auto, tag _v2) and the small-hole check on e2e
batch 2 (tag _batch2_b4), all with fill.large_models = b4.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of the data kernel, the Stage 2 checkpoint kernels, the
frozen Stage 1 kernels (B2) and floorplan-stage2-b4 (the B4 checkpoints).
Output: /kaggle/working/stage2/ (b4_val.md / .csv, large_fill_compare_v2_*.md / .csv / .png, e2e_val_batch2_b4.*).
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


def use_b4():
    """fill.large_models = b4 in this kernel's copy of configs/stage2/pipeline.yaml."""
    p = f"{CODE}/configs/stage2/pipeline.yaml"
    s = open(p, encoding="utf-8").read().replace("  large_models: current ", "  large_models: b4      ")
    open(p, "w", encoding="utf-8").write(s)


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import run
    from src.stage2.kaggle_cpu import finish, setup
    data = setup()
    use_b4()
    try:
        run(f"{sys.executable} -m src.stage2.complete.b4_val --data {data}")
        for mode in ("manual", "auto"):
            run(f"{sys.executable} -m src.stage2.pipeline.large_fill_compare --zip {data}/parser_val.zip --mode {mode} "
                f"--tag _v2_{mode}")
        run(f"{sys.executable} -m src.stage2.pipeline.e2e_val --zip {data}/parser_val.zip --batch 2 --tag _batch2_b4")
    finally:
        finish()

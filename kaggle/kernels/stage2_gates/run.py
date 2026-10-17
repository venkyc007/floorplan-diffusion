"""Kaggle kernel (GPU T4): constraint weight calibration + gates G3 and G4 for K1.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels
venkatesh007c/floorplan-stage2-data (completion_train.zip) and
venkatesh007c/floorplan-finetune-constraints (Stage 1 best.pt for the warm start).
Output /kaggle/working/gates/: constraint_weights.md/.yaml, gate_g3.md, gate_g4.md (configs are NOT
changed here: the user reviews the weights first).
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
OUT = "/kaggle/working/gates"


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
    from src.kaggle_kernel import find, run
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    os.makedirs(OUT, exist_ok=True)
    zip_ = find("completion_train.zip")[0]
    ckpt = find("runs/finetune_constraints/checkpoints/best.pt")[0]
    print("data", zip_, "| stage 1", ckpt, flush=True)
    try:
        run(f"{sys.executable} -m src.stage2.complete.calibrate --zip {zip_} --stage1 {ckpt} --batches 50 --batch 8")
        run(f"{sys.executable} -m src.stage2.complete.gates34 g3 --zip {zip_}")
        run(f"{sys.executable} -m src.stage2.complete.gates34 g4 --zip {zip_} --batch 32 --steps 10")
    finally:
        for f in ("outputs/stage2/constraint_weights.md", "outputs/stage2/constraint_weights.yaml", "outputs/stage2/gate_g3.md", "outputs/stage2/gate_g4.md",
                  "configs/stage2/train16_constrained.yaml", "configs/stage2/train16_control.yaml"):
            if os.path.exists(f"{CODE}/{f}"):
                shutil.copy(f"{CODE}/{f}", OUT)

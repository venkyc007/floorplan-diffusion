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
    # G3 variant re-run (user, 2026-10-17); calibration + G4 are done (outputs/stage2/constraint_weights.md, gate_g4*.md)
    steps = [f"{sys.executable} -m src.stage2.complete.gates34 g3v --zip {zip_}"]
    failed = []
    try:
        for i, cmd in enumerate(steps):
            try:
                run(cmd)
            except Exception as e:          # each gate independent; calibration must succeed for G3
                print("FAILED:", cmd, e, flush=True)
                failed.append(cmd)
                if i == 0:
                    break
    finally:
        for f in ("outputs/stage2/constraint_weights.md", "outputs/stage2/constraint_weights.yaml", "outputs/stage2/gate_g3.md", "outputs/stage2/gate_g4.md", "outputs/stage2/gate_g4_K48.md", "outputs/stage2/gate_g3_variants.md", "outputs/stage2/gate_g3_choice.json",
                  "configs/stage2/train16_constrained.yaml", "configs/stage2/train16_control.yaml"):
            if os.path.exists(f"{CODE}/{f}"):
                shutil.copy(f"{CODE}/{f}", OUT)
    if failed:
        raise SystemExit(f"failed: {failed}")

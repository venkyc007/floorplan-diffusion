"""Kaggle kernel: fine-tune diffusion run 1 with structural constraints ON, on all GPUs
(training only). Starts from run 1's best.pt (EMA weights, attached as the output
of kernel venkatesh007c/floorplan-diffusion). See docs/kaggle_workflow.md.

Inputs (private datasets): venkatesh007c/floorplan-data, venkatesh007c/floorplan-code.
Outputs in /kaggle/working/runs/finetune_constraints: checkpoints/{last,best}.pt (EMA weights
in best.pt), train_log.csv (loss + samples/s + GPU count), val_log.csv,
samples/, config.yaml, gpu_usage.csv (nvidia-smi every 30 s).
Resume: attach an earlier session's output as an input; training continues
from its runs/finetune_constraints/checkpoints/last.pt.
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"


def copy_code():
    """Find the code dataset (unpacked by Kaggle, or code.zip) and copy it to CODE."""
    hits = [h for h in glob.glob("/kaggle/input/**/src/stage1/train.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import WORK, check_val_benchmark, find, gpu_monitor, restore_previous_run, setup, train
    RUN = f"{WORK}/runs/finetune_constraints"
    setup()
    check_val_benchmark()
    restore_previous_run(RUN, "finetune_constraints")
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    try:
        init = find("runs/diffusion/checkpoints/best.pt")[0]   # run 1 (EMA weights)
        train("configs/stage1/finetune_constraints.yaml", RUN, f"--init {init}")
    finally:
        monitor.terminate()

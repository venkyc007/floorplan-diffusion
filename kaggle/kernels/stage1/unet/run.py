"""Kaggle kernel: train the U-Net baseline on all GPUs, then score its best
checkpoint on the full val benchmark. See docs/kaggle_workflow.md.

Inputs (private datasets): venkatesh007c/floorplan-data, venkatesh007c/floorplan-code.
Outputs in /kaggle/working/runs/unet: checkpoints/{last,best}.pt, train_log.csv
(loss + samples/s + GPU count), val_log.csv, samples/, config.yaml,
eval_val.csv/.md, gpu_usage.csv (nvidia-smi every 30 s).
Resume: attach an earlier session's output as an input; training continues
from its runs/unet/checkpoints/last.pt.
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
    from src.kaggle_kernel import (WORK, check_val_benchmark, gpu_monitor, restore_previous_run,
                                   run, setup, train)
    RUN = f"{WORK}/runs/unet"
    setup()
    check_val_benchmark()
    restore_previous_run(RUN, "unet")
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    try:
        train("configs/stage1/train_unet.yaml", RUN)
    finally:
        monitor.terminate()
    run(f"{sys.executable} -m src.stage1.evaluate --checkpoint {RUN}/checkpoints/best.pt --benchmark val")

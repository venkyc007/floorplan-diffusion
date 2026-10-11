"""Kaggle kernel: matched-sample U-Net (fairness check). Trains the U-Net
baseline from scratch on all GPUs for ~1.92 M samples (configs/train_unet_matched.yaml),
then evaluates its best checkpoint on the full val benchmark and keeps the
per-case predictions. See docs/kaggle_workflow.md.

Inputs (private datasets): venkatesh007c/floorplan-data, venkatesh007c/floorplan-code.
Outputs in /kaggle/working:
  runs/unet_matched/        checkpoints, train_log.csv, val_log.csv, samples/, gpu_usage.csv
  full_eval_unet_matched/   full_eval.csv / full_eval.md (all metrics, all breakdowns)
  preds/unet_matched.npz    per-case predicted label maps (uint8, compressed) + plan_ids
Resume: attach an earlier session's output as an input (runs/unet_matched/checkpoints/last.pt).
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"


def copy_code():
    """Find the code dataset (unpacked by Kaggle, or code.zip) and copy it to CODE."""
    hits = [h for h in glob.glob("/kaggle/input/**/src/train.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import WORK, check_val_benchmark, gpu_monitor, restore_previous_run, run, setup, train
    RUN = f"{WORK}/runs/unet_matched"
    setup()
    check_val_benchmark()
    restore_previous_run(RUN, "unet_matched")
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    try:
        train("configs/train_unet_matched.yaml", RUN)
    finally:
        monitor.terminate()
    common = f"--only unet_matched --preds /tmp/preds"
    run(f"{sys.executable} -m src.full_eval {common} --ckpt-unet-matched {RUN}/checkpoints/best.pt "
        f"--keep-preds {WORK}/preds")
    run(f"{sys.executable} -m src.full_eval --score {common} --out {WORK}/full_eval_unet_matched")

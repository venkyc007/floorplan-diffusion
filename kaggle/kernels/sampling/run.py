"""Kaggle kernel: sampling experiment for diffusion run 1 (src/sampling_experiment.py)
on the 256 fixed val cases, one half of the work per GPU. See docs/kaggle_workflow.md.

Inputs: datasets venkatesh007c/floorplan-data + floorplan-code, and the OUTPUTS of
kernels venkatesh007c/floorplan-diffusion and floorplan-unet (their best.pt).
Outputs in /kaggle/working/sampling: sampling_experiment.csv / sampling_table.md,
sampling_experiment.png, part*_labels.npz, part*_seconds.json, gpu_usage.csv.
"""

import glob
import os
import shutil
import subprocess
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
    from src.kaggle_kernel import WORK, check_val_benchmark, find, gpu_count, gpu_monitor, run, setup
    OUT = f"{WORK}/sampling"
    setup()
    check_val_benchmark()
    diffusion = find("runs/diffusion/checkpoints/best.pt")[0]
    unet = find("runs/unet/checkpoints/best.pt")[0]
    print("diffusion checkpoint:", diffusion, "| U-Net checkpoint:", unet, flush=True)
    monitor = gpu_monitor(f"{OUT}/gpu_usage.csv")
    n = max(1, min(2, gpu_count()))
    base = f"{sys.executable} -m src.sampling_experiment --diffusion {diffusion} --unet {unet} --out {OUT}"
    try:
        # One process per GPU, each on half of the settings.
        procs = [subprocess.Popen(f"{base} --part {k} --parts {n}", shell=True, cwd=CODE,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTHONUNBUFFERED": "1"})
                 for k in range(n)]
        codes = [p.wait() for p in procs]
        print("part exit codes:", codes, flush=True)
    finally:
        monitor.terminate()
    run(f"{sys.executable} -m src.sampling_experiment --report --out {OUT}")
    if any(codes):
        raise SystemExit(f"a part failed: {codes}")

"""Kaggle kernel: timing run on all GPUs. See docs/kaggle_workflow.md.

Runs the REAL training command (torchrun on every GPU) for a few hundred
steps per model, so the measured speed includes the multi-GPU overhead.

Inputs (private datasets): venkatesh007c/floorplan-data, venkatesh007c/floorplan-code.
Outputs in /kaggle/working: ddp_speed.json (summary), benchmark_check.json,
timing/<model>/ (train_log.csv, val_log.csv, gpu_usage.csv, ...).
"""

import csv
import glob
import json
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"
STEPS = 300


def copy_code():
    """Find the code dataset (unpacked by Kaggle, or code.zip) and copy it to CODE."""
    hits = [h for h in glob.glob("/kaggle/input/**/src/train.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


def summarize(run_dir):
    import yaml
    rows = list(csv.DictReader(open(f"{run_dir}/train_log.csv")))[1:]   # skip warm-up row
    samples = sum(float(r["samples_per_s"]) for r in rows) / len(rows)
    gpus = int(rows[-1]["gpus"])                                         # processes used
    per_step = yaml.safe_load(open(f"{run_dir}/config.yaml"))["train"]["batch_size"] * gpus
    val = list(csv.DictReader(open(f"{run_dir}/val_log.csv")))[-1]
    usage = list(csv.reader(open(f"{run_dir}/gpu_usage.csv")))[1:]
    busy = {}
    for r in usage:   # timestamp, index, utilization.gpu [%], memory.used [MiB]
        busy.setdefault(r[1].strip(), []).append(float(r[2].strip().split()[0]))
    return {"gpus": gpus, "samples_per_s": round(samples, 1), "batch_per_step": per_step,
            "steps_per_s": round(samples / per_step, 3), "val_seconds": float(val["val_seconds"]),
            "mean_gpu_utilization_pct": {k: round(sum(v) / len(v), 1) for k, v in busy.items()}}


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import WORK, check_val_benchmark, gpu_monitor, setup, train
    setup()
    check_val_benchmark()
    results = {}
    for kind in ("unet", "diffusion"):
        out = f"{WORK}/timing/{kind}"
        monitor = gpu_monitor(f"{out}/gpu_usage.csv")
        try:
            train(f"configs/train_{kind}.yaml", out, f"--max-steps {STEPS}")
        finally:
            monitor.terminate()
        results[kind] = summarize(out)
        print(kind, json.dumps(results[kind]), flush=True)
    json.dump(results, open(f"{WORK}/ddp_speed.json", "w"), indent=2)
    print(json.dumps(results, indent=2))

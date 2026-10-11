"""Kaggle kernel (Day 6, B.3 + B.4): predictions of every GPU method on the
full val benchmark, SAVED per case (compressed) so all metrics (old +
architectural), post-processing and statistics can be computed later on a CPU.
One process per GPU, about half of the work each. See docs/kaggle_workflow.md.

Inputs: datasets venkatesh007c/floorplan-data + floorplan-code, and the OUTPUTS
(best.pt) of kernels floorplan-diffusion, floorplan-unet,
floorplan-finetune-constraints and floorplan-finetune-control.
Outputs in /kaggle/working: preds/<setting>.npz (uint8 label maps + plan_ids), gpu_usage.csv.
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
PREDS = "/tmp/preds"     # predicted label maps (large) stay out of the kernel output


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
    OUT = WORK
    # Balanced by measured cost (s/case on a T4): ~1.1-1.3 s per case per GPU.
    PARTS = ["ft_constraints_ddim50,ft_constraints_refine800,ft_constraints_refine200,unet",
             "run1_refine600,ft_constraints_refine600,ft_control_refine600,ft_constraints_refine400"]
    setup()
    check_val_benchmark()
    ckpt = {name: find(f"runs/{name}/checkpoints/best.pt")[0]
            for name in ("diffusion", "unet", "finetune_constraints", "finetune_control")}
    print("checkpoints:", ckpt, flush=True)
    args = (f"--preds {PREDS} --ckpt-run1 {ckpt['diffusion']} --ckpt-unet {ckpt['unet']} "
            f"--ckpt-ft-constraints {ckpt['finetune_constraints']} --ckpt-ft-control {ckpt['finetune_control']}")
    monitor = gpu_monitor(f"{OUT}/gpu_usage.csv")
    n = max(1, min(2, gpu_count()))
    parts = PARTS if n == 2 else [",".join(PARTS)]
    try:
        procs = [subprocess.Popen(f"{sys.executable} -m src.full_eval --only {parts[k]} {args} --keep-preds {WORK}/preds",
                                  shell=True, cwd=CODE,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTHONUNBUFFERED": "1"})
                 for k in range(n)]
        codes = [p.wait() for p in procs]
        print("part exit codes:", codes, flush=True)
    finally:
        monitor.terminate()
    if any(codes):
        raise SystemExit(f"a part failed: {codes}")

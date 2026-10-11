"""Kaggle kernel (Day 7, selection experiment): predictions of the new
settings on the full val benchmark, SAVED per case (compressed); all scoring
runs later on a CPU (python -m src.selection). No training.

Settings (see src/full_eval.py):
  hybrid_refine{400,600,700,800,900}  matched U-Net prediction as the sketch,
                                      noised to t0, finished by the constraint fine-tune
  ft_constraints_refine{700,900}      constraint fine-tune refined from its own one-shot

Inputs: datasets venkatesh007c/floorplan-data + floorplan-code, and the OUTPUTS of
kernels floorplan-finetune-constraints (best.pt) and floorplan-unet-matched
(preds/unet_matched.npz = the exact U-Net predictions that were scored).
Outputs in /kaggle/working: preds/<setting>.npz, gpu_usage.csv.
One process per GPU, about half of the work each. See docs/kaggle_workflow.md.
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
PREDS = "/tmp/preds"     # uncompressed label maps (large) stay out of the kernel output


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
    from src.kaggle_kernel import WORK, check_val_benchmark, find, gpu_count, gpu_monitor, setup
    # Balanced by measured cost (s/case on a T4, about proportional to t0): ~1.3-1.5 s per case per GPU.
    PARTS = ["hybrid_refine900,ft_constraints_refine900,hybrid_refine600",
             "hybrid_refine800,ft_constraints_refine700,hybrid_refine700,hybrid_refine400"]
    setup()
    check_val_benchmark()
    ckpt = find("runs/finetune_constraints/checkpoints/best.pt")[0]
    sketch = find("preds/unet_matched.npz")[0]
    print("checkpoint:", ckpt, "sketch:", sketch, flush=True)
    args = f"--preds {PREDS} --ckpt-ft-constraints {ckpt} --sketch {sketch}"
    monitor = gpu_monitor(f"{WORK}/gpu_usage.csv")
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

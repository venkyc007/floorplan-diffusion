"""Kaggle kernel floorplan-stage1-test (GPU T4 x2): the ONE final Stage 1 test run (frozen Day 7 configurations).

Predictions on the full TEST benchmark, saved per case (compressed); scoring runs afterwards in the CPU kernel
floorplan-stage1-test-score (python -m src.selection --benchmark test). No training, no tuning.
  1. matched U-Net (GPU 0) -> preds/unet_matched.npz = the sketch of the hybrids
  2. GPU 0: hybrid_refine700, hybrid_refine400      GPU 1: hybrid_refine600, ft_constraints_refine600
The size-adaptive hybrid (t0 = 400 / 600 / 700 per size bin) is assembled from the three hybrids at scoring time.

Inputs: datasets venkatesh007c/floorplan-code + venkatesh007c/floorplan-test-benchmark (checksum checked against
configs/benchmark_checksums.json); outputs of kernels floorplan-unet-matched and floorplan-finetune-constraints.
Output: /kaggle/working/preds/<setting>.npz, timing json, gpu_usage.csv.
"""

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
NPY = "/tmp/npy"


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
    from src.kaggle_kernel import WORK, find, gpu_count, gpu_monitor, run
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    bdir = f"{CODE}/data/benchmarks"
    os.makedirs(bdir, exist_ok=True)
    for f in ("test_benchmark.npz", "test_benchmark.csv"):
        shutil.copy(find(f)[0], f"{bdir}/{f}")
    want = json.load(open(f"{CODE}/configs/benchmark_checksums.json"))["test"]
    got = hashlib.sha256(open(f"{bdir}/test_benchmark.npz", "rb").read()).hexdigest()
    print("test benchmark sha256", got, "MATCHES" if got == want else "DIFFERS (stop)", flush=True)
    if got != want:
        raise SystemExit("test benchmark differs from the local one")
    unet = find("runs/unet_matched/checkpoints/best.pt")[0]
    ft = find("runs/finetune_constraints/checkpoints/best.pt")[0]
    print("checkpoints:", unet, ft, flush=True)
    keep = f"{WORK}/preds"
    monitor = gpu_monitor(f"{WORK}/gpu_usage.csv")
    try:
        run(f"{sys.executable} -m src.full_eval --benchmark test --only unet_matched --preds {NPY} --keep-preds {keep} "
            f"--ckpt-unet-matched {unet}")
        parts = ["hybrid_refine700,hybrid_refine400", "hybrid_refine600,ft_constraints_refine600"]
        n = max(1, min(2, gpu_count()))
        if n == 1:
            parts = [",".join(parts)]
        args = f"--benchmark test --preds {NPY} --keep-preds {keep} --ckpt-ft-constraints {ft} --sketch {keep}/unet_matched.npz"
        procs = [subprocess.Popen(f"{sys.executable} -m src.full_eval --part {k} --only {parts[k]} {args}", shell=True,
                                  cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTHONUNBUFFERED": "1"})
                 for k in range(n)]
        codes = [p.wait() for p in procs]
        print("part exit codes:", codes, flush=True)
    finally:
        monitor.terminate()
        for f in glob.glob(f"{NPY}/timing_part*.json"):
            shutil.copy(f, WORK)
    if any(codes):
        raise SystemExit(f"a part failed: {codes}")

"""Kaggle kernel K1 (GPU T4 x2): 16-class diffusion, constrained (GPU0) vs control (GPU1), side by side.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels
venkatesh007c/floorplan-stage2-data (completion_{train,val}.zip) and
venkatesh007c/floorplan-finetune-constraints (Stage 1 best.pt, warm start).
If an earlier K1 output is attached (runs/k1_*/checkpoints/last.pt), both runs resume.
Output: /kaggle/working/runs/k1_constrained/, /kaggle/working/runs/k1_control/ (+ gpu_usage.csv).
Each run stops itself at time_limit_hours (10.5) and saves; last.pt every 30 min.
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/k1data"
RUNS = "/kaggle/working/runs"


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
    from src.kaggle_kernel import find, gpu_monitor, run
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    os.makedirs(DATA, exist_ok=True)
    for f in ("completion_train.zip", "completion_val.zip"):
        shutil.copy(find(f)[0], f"{DATA}/{f}")
    init = find("runs/finetune_constraints/checkpoints/best.pt")[0]
    for name in ("k1_constrained", "k1_control"):
        last = find(f"runs/{name}/checkpoints/last.pt")
        if last and not os.path.exists(f"{RUNS}/{name}/checkpoints/last.pt"):
            shutil.copytree(os.path.dirname(os.path.dirname(last[0])), f"{RUNS}/{name}", dirs_exist_ok=True)
            print("resuming", name, "from", last[0], flush=True)
    os.makedirs(RUNS, exist_ok=True)
    monitor = gpu_monitor(f"{RUNS}/gpu_usage.csv")
    procs = []
    try:
        for gpu, (name, cfg) in enumerate([("k1_constrained", "train16_constrained"), ("k1_control", "train16_control")]):
            log = open(f"{RUNS}/{name}.log", "a")
            cmd = (f"{sys.executable} -m src.stage2.complete.train16 --config configs/stage2/{cfg}.yaml "
                   f"--data {DATA} --out {RUNS}/{name} --init {init}")
            print(f"GPU{gpu}: {cmd}", flush=True)
            procs.append(subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}))
        codes = [p.wait() for p in procs]
        print("exit codes:", codes, flush=True)
    finally:
        monitor.terminate()
        for name in ("k1_constrained", "k1_control"):
            p = f"{RUNS}/{name}.log"
            if os.path.exists(p):
                print(f"===== tail of {name}.log", flush=True)
                print("".join(open(p).readlines()[-15:]), flush=True)
    if any(codes):
        raise SystemExit(f"a run failed: {codes}")

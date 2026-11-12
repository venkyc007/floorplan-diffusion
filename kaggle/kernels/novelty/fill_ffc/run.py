"""Kaggle kernel floorplan-novelty-fill-ffc (GPU T4 x2): the large-hole FILL models WITH Fourier context blocks: GPU0 the U-Net, GPU1 the
diffusion model, both warm-started from B4 (src/novelty/train_fill.py, configs/novelty/fill_*.yaml).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-novelty-data (the large-omission
set) and floorplan-stage2-b4 (B4 checkpoints).
Output: /kaggle/working/runs/<run name>/ (checkpoints, logs).
A watchdog stops anything still running 3.3 h after the kernel start (each run stops itself earlier).
"""

import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

CODE = "/tmp/code"
RUNS = "/kaggle/working/runs"
START = time.time()
WATCHDOG_H = 3.3


def copy_code():
    hits = [h for h in glob.glob("/kaggle/input/**/src/utils.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


def run_jobs(jobs):
    """One job per GPU, in parallel; logs to RUNS/<name>.log; watchdog; tails printed at the end."""
    os.makedirs(RUNS, exist_ok=True)
    procs = []
    for gpu, (name, args) in enumerate(jobs):
        log = open(f"{RUNS}/{name}.log", "a")
        print(f"GPU{gpu}: {name}: {args}", flush=True)
        procs.append(subprocess.Popen(f"{sys.executable} {args}", shell=True, cwd=CODE, stdout=log,
                                      stderr=subprocess.STDOUT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}))
    while any(p.poll() is None for p in procs):
        if (time.time() - START) / 3600 >= WATCHDOG_H:
            print("watchdog: time budget reached, terminating", flush=True)
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            time.sleep(30)
            break
        time.sleep(30)
    print("exit codes:", [p.poll() for p in procs], f"kernel time {(time.time() - START) / 3600:.2f} h", flush=True)
    for name, _ in jobs:
        print(f"===== tail of {name}.log", flush=True)
        print("".join(open(f"{RUNS}/{name}.log").readlines()[-25:]), flush=True)


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import find, run
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    train = " ".join(sorted(find("large_train_*.zip")))
    val = " ".join(sorted(find("large_tune_*.zip")))
    jobs = []
    for kind in ("unet", "diffusion"):
        init = find(f"runs/b4_{kind}/checkpoints/best.pt")[0]
        jobs.append((f"fill_{kind}_ffc", f"-m src.novelty.train_fill --config configs/novelty/fill_{kind}.yaml "
                     f"--train {train} --val {val} --out {RUNS}/fill_{kind}_ffc --init16 {init}"))
    run_jobs(jobs)

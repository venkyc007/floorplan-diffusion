"""Kaggle kernel B4 (GPU T4 x2, user 2026-10-21): fine-tune the 16-class U-Net (GPU0) and the constrained diffusion
(GPU1) on 256 x 256 windows with 50% large holes, warm-started from the CURRENT checkpoints (k2a_unet, k1_constrained).

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (completion zips),
floorplan-stage2-k2a and floorplan-stage2-k1 (current checkpoints, read only). If an earlier B4 output is attached,
both runs resume from their last.pt.
Output: /kaggle/working/runs/b4_unet/, /kaggle/working/runs/b4_diffusion/ (new names; + logs, gpu_usage.csv).
Hard budget (user): each run stops itself after 1.65 h of training (configs/stage2/train16_b4_*.yaml) and saves;
a watchdog here terminates anything still running at 1.85 h after the kernel start (last.pt is at most 15 min old,
best.pt is written at every validation), so the kernel stays < 1.9 h on the clock and < 3.8 GPU h.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/b4data"
RUNS = "/kaggle/working/runs"
START = time.time()
WATCHDOG_H = 1.85


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
    unet0 = find("runs/k2a_unet/checkpoints/best.pt")[0]
    diff0 = find("runs/k1_constrained/checkpoints/best.pt")[0]
    jobs = [("b4_unet", f"-m src.stage2.complete.train16 --config configs/stage2/train16_b4_unet.yaml "
                        f"--data {DATA} --out {RUNS}/b4_unet --init16 {unet0}"),
            ("b4_diffusion", f"-m src.stage2.complete.train16 --config configs/stage2/train16_b4_diffusion.yaml "
                             f"--data {DATA} --out {RUNS}/b4_diffusion --init16 {diff0}")]
    for name, _ in jobs:
        last = find(f"runs/{name}/checkpoints/last.pt")
        if last and not os.path.exists(f"{RUNS}/{name}/checkpoints/last.pt"):
            shutil.copytree(os.path.dirname(os.path.dirname(last[0])), f"{RUNS}/{name}", dirs_exist_ok=True)
            print("resuming", name, "from", last[0], flush=True)
    os.makedirs(RUNS, exist_ok=True)
    monitor = gpu_monitor(f"{RUNS}/gpu_usage.csv")
    procs, codes = [], []
    try:
        for gpu, (name, args) in enumerate(jobs):
            log = open(f"{RUNS}/{name}.log", "a")
            print(f"GPU{gpu}: {name}", flush=True)
            procs.append(subprocess.Popen(f"{sys.executable} {args}", shell=True, cwd=CODE, stdout=log,
                                          stderr=subprocess.STDOUT,
                                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}))
        while any(p.poll() is None for p in procs):
            if (time.time() - START) / 3600 >= WATCHDOG_H:
                print("watchdog: hard budget reached, terminating", flush=True)
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
                time.sleep(30)
                for p in procs:
                    if p.poll() is None:
                        p.kill()
                break
            time.sleep(30)
        codes = [p.wait() for p in procs]
        print("exit codes:", codes, f"kernel time {(time.time() - START) / 3600:.2f} h", flush=True)
    finally:
        monitor.terminate()
        for name, _ in jobs:
            p = f"{RUNS}/{name}.log"
            if os.path.exists(p):
                print(f"===== tail of {name}.log", flush=True)
                print("".join(open(p).readlines()[-15:]), flush=True)

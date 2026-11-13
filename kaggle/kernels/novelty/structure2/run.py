"""Kaggle kernel floorplan-novelty-structure2 (GPU T4 x2): the STRUCTURE net with the STROKE map as a 4th extra input
(src/novelty/structure.py: parsed map + continuation rays + ink map + stroke map -> missing walls), two seeds,
one per GPU, 2.6 h each.

Inputs: code dataset venkatesh007c/floorplan-code; output of kernel floorplan-novelty-data2 (the bigger large-omission
set with ink maps: large9_train_*.zip, large9_tune_*.zip).
Output: /kaggle/working/runs/<run name>/ (checkpoints, logs).
A watchdog stops anything still running 3.1 h after the kernel start (each run stops itself earlier).
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
WATCHDOG_H = 3.1


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
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    train = " ".join(sorted(find("large9_train_*.zip")))
    val = " ".join(sorted(find("large9_tune_*.zip")))
    base = f"-m src.novelty.structure --config configs/novelty/structure.yaml --train {train} --val {val}"
    run_jobs([(f"struct4_s{seed}", f"{base} --out {RUNS}/struct4_s{seed} --seed {seed} --hours 2.6") for seed in (0, 1)])

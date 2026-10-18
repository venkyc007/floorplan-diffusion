"""Kaggle kernel K2a (GPU T4 x2): GPU0 16-class U-Net (sketch model), GPU1 missing-component detector.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels
venkatesh007c/floorplan-stage2-data (completion_{train,val}.zip) and
venkatesh007c/floorplan-unet-matched (Stage 1 matched U-Net best.pt, warm start).
If an earlier K2a output is attached, both runs resume from their last.pt.
Output: /kaggle/working/runs/k2a_unet/, /kaggle/working/runs/k2a_detector/ (+ logs, gpu_usage.csv).
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/k2data"
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
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    os.makedirs(DATA, exist_ok=True)
    for f in ("completion_train.zip", "completion_val.zip"):
        shutil.copy(find(f)[0], f"{DATA}/{f}")
    init = find("runs/unet_matched/checkpoints/best.pt")[0]
    jobs = [("k2a_unet", f"-m src.stage2.complete.train16 --config configs/stage2/train16_unet.yaml "
                         f"--data {DATA} --out {RUNS}/k2a_unet --init {init}"),
            ("k2a_detector", f"-m src.stage2.detect.train_detector --config configs/stage2/detector.yaml "
                             f"--data {DATA} --out {RUNS}/k2a_detector")]
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
        codes = [p.wait() for p in procs]
        print("exit codes:", codes, flush=True)
    finally:
        monitor.terminate()
        for name, _ in jobs:
            p = f"{RUNS}/{name}.log"
            if os.path.exists(p):
                print(f"===== tail of {name}.log", flush=True)
                print("".join(open(p).readlines()[-15:]), flush=True)
    if any(codes):
        raise SystemExit(f"a run failed: {codes}")

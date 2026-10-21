"""Helpers used by the Kaggle kernel scripts in kaggle/kernels/*/run.py.

A kernel's run.py first copies the code to CODE (it can't import anything
before that), then imports this module for everything else:

    setup()                  data into place, missing packages installed, GPU info
    check_val_benchmark()    rebuild the val benchmark from its seed, compare checksums
    train(config, out)       train on all GPUs (torchrun if more than one)
    gpu_monitor(path)        log every GPU's utilisation/memory every 30 s
    run(cmd)                 run a shell command in CODE, stop on error
"""

import glob
import json
import os
import shutil
import subprocess
import sys

CODE = "/tmp/code"
WORK = "/kaggle/working"


def find(name, must_contain=""):
    return sorted(h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if must_contain in h)


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=CODE, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def setup():
    os.makedirs(f"{CODE}/data/processed", exist_ok=True)
    for f in ("train.npz", "val.npz"):
        shutil.copy(find(f, "floorplan-data")[0], f"{CODE}/data/processed/{f}")
    missing = []
    for module, package in [("diffusers", "diffusers"), ("cv2", "opencv-python-headless"),
                            ("shapely", "shapely"), ("skimage", "scikit-image"), ("yaml", "pyyaml")]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        run(f"{sys.executable} -m pip install -q " + " ".join(missing))
    run(f"{sys.executable} -c \"import torch, diffusers; print('torch', torch.__version__, "
        f"'| diffusers', diffusers.__version__, '| GPUs', torch.cuda.device_count())\"")
    run("nvidia-smi --query-gpu=index,name,memory.total --format=csv")


def check_val_benchmark():
    """Rebuild the val benchmark from its seed; compare with the local checksum."""
    sums = f"{CODE}/configs/stage1/benchmark_checksums.json"
    expected = json.load(open(sums))["val"]
    run(f"{sys.executable} -m src.stage1.data.make_benchmark --splits val")
    got = json.load(open(sums))["val"]           # make_benchmark rewrites it
    result = {"expected": expected, "kaggle": got, "match": expected == got}
    json.dump(result, open(f"{WORK}/benchmark_check.json", "w"), indent=2)
    print("VAL BENCHMARK CHECKSUM", "MATCHES" if result["match"] else "DIFFERS", result, flush=True)
    if not result["match"]:
        raise SystemExit("val benchmark differs from the local one")


def gpu_monitor(path):
    """Start logging every GPU's utilisation, memory, SM clock (current and max),
    temperature and power to a CSV every 30 s (to spot clock throttling).
    Returns the process. The first 4 columns keep their order (timing summaries)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = "timestamp,index,utilization.gpu,memory.used,clocks.sm,clocks.max.sm,temperature.gpu,power.draw"
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv", "-l", "30"],
                            stdout=open(path, "w"), stderr=subprocess.DEVNULL)


def train(config, out, extra=""):
    """Train on every available GPU: torchrun (one process per GPU) if > 1."""
    n = gpu_count()
    if n > 1:
        launcher = f"{sys.executable} -m torch.distributed.run --standalone --nproc_per_node={n}"
    else:
        launcher = sys.executable
    run(f"{launcher} -m src.stage1.train --config {config} --out {out} {extra}".strip())


def restore_previous_run(run_dir, name):
    """If an earlier session's output is attached as an input, continue from its last.pt."""
    last = find(f"runs/{name}/checkpoints/last.pt")
    if last and not os.path.exists(f"{run_dir}/checkpoints/last.pt"):
        previous = os.path.dirname(os.path.dirname(last[0]))
        shutil.copytree(previous, run_dir, dirs_exist_ok=True)
        print(f"Resuming from {previous}", flush=True)

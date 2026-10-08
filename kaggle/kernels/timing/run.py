"""Kaggle kernel: timing run (GPU speed of both models). See docs/kaggle_workflow.md.

Inputs (private datasets): venkatesh007c/floorplan-data, venkatesh007c/floorplan-code.
Outputs in /kaggle/working: speed_results.json, benchmark_check.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"          # working copy of the code (not saved as output)
WORK = "/kaggle/working"    # everything here is saved as kernel output


def find(name, must_contain=""):
    hits = [h for h in glob.glob(f"/kaggle/input/**/{name}", recursive=True) if must_contain in h]
    return sorted(hits)


def setup():
    """Copy code + data into CODE, install missing packages."""
    train_py = find("src/train.py", "floorplan-code")
    if train_py:   # Kaggle unpacked code.zip into folders
        shutil.copytree(os.path.dirname(os.path.dirname(train_py[0])), CODE, dirs_exist_ok=True)
    else:
        zipfile.ZipFile(find("code.zip", "floorplan-code")[0]).extractall(CODE)
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
        f"'| diffusers', diffusers.__version__, '| cuda', torch.cuda.is_available())\"")
    run("nvidia-smi --query-gpu=name,memory.total --format=csv")


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=CODE if os.path.isdir(CODE) else None, check=True,
                   env={**os.environ, "PYTHONUNBUFFERED": "1"})


def check_val_benchmark():
    """Rebuild the val benchmark from its seed; compare with the local checksum."""
    sums = f"{CODE}/configs/benchmark_checksums.json"
    expected = json.load(open(sums))["val"]
    run(f"{sys.executable} -m src.data.make_benchmark --splits val")
    got = json.load(open(sums))["val"]           # make_benchmark rewrites it
    result = {"expected": expected, "kaggle": got, "match": expected == got}
    json.dump(result, open(f"{WORK}/benchmark_check.json", "w"), indent=2)
    print("VAL BENCHMARK CHECKSUM", "MATCHES" if result["match"] else "DIFFERS", result, flush=True)
    if not result["match"]:
        raise SystemExit("val benchmark differs from the local one")


if __name__ == "__main__":
    setup()
    check_val_benchmark()
    run(f"{sys.executable} -m src.speed_test --steps 200 --out {WORK}/speed_results.json")

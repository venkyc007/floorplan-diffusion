"""Kaggle kernel (GPU T4 x2): parser v2, two variants side by side (one per GPU), then v1 / v2A / v2B compared.

  GPU 0: v2A "no text" (text erased on 100% of crops)    GPU 1: v2B "mixed" (50%)
  both fine-tuned from parser v1 with the local constraints (src/stage2/parse/train_v2.py, configs/stage2/parser_v2.yaml);
  then src/stage2/parse/eval_v2.py on the 159 held-out val drawings -> parser_v2.md.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (parser zips, stats)
and floorplan-stage2-parser-v1 (best.pt).
Output: /kaggle/working/runs/parser_v2A|B/ (checkpoints, logs, constraint_weights.yaml), /kaggle/working/stage2/parser_v2.md.
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/parser"
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
    for split in ("train", "val"):
        z = find(f"parser_{split}.zip")[0]
        zipfile.ZipFile(z).extractall(f"{DATA}/{split}")
        print(split, len(os.listdir(f"{DATA}/{split}")) // 2, "plans from", z, flush=True)
    stats = find("report/stats.npz")[0]
    v1 = find("runs/parser_v1/checkpoints/best.pt")[0]
    split = f"{CODE}/configs/stage2/val_split"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    monitor = gpu_monitor(f"{RUNS}/gpu_usage.csv")
    procs = []
    for gpu, var in ((0, "A"), (1, "B")):
        out = f"{RUNS}/parser_v2{var}"
        os.makedirs(out, exist_ok=True)
        cmd = (f"{sys.executable} -m src.stage2.parse.train_v2 --config configs/stage2/parser_v2.yaml --variant {var} "
               f"--data {DATA} --init {v1} --out {out} --freq {stats} --tuning {split}/tuning_ids.txt")
        print(f"$ GPU {gpu}: {cmd}", flush=True)
        procs.append(subprocess.Popen(cmd, shell=True, cwd=CODE, stdout=open(f"{out}/stdout.log", "w"),
                                      stderr=subprocess.STDOUT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}))
    codes = [p.wait() for p in procs]
    monitor.terminate()
    for var in "AB":
        print(f"--- v2{var} log tail ---", flush=True)
        print("".join(open(f"{RUNS}/parser_v2{var}/stdout.log").readlines()[-15:]), flush=True)
    print("exit codes", codes, flush=True)
    os.makedirs(f"{CODE}/outputs/stage2", exist_ok=True)
    try:
        run(f"{sys.executable} -m src.stage2.parse.eval_v2 --val {DATA}/val --heldout {split}/heldout_ids.txt "
            f"--v1 {v1} --v2a {RUNS}/parser_v2A/checkpoints/best.pt --v2b {RUNS}/parser_v2B/checkpoints/best.pt")
    finally:
        os.makedirs("/kaggle/working/stage2", exist_ok=True)
        if os.path.exists(f"{CODE}/outputs/stage2/parser_v2.md"):
            shutil.copy(f"{CODE}/outputs/stage2/parser_v2.md", "/kaggle/working/stage2/")

"""Kaggle kernel (GPU T4 x2): train parser v1 (src/stage2/parse/train.py, configs/stage2/parser_v1.yaml).

Inputs: code dataset venkatesh007c/floorplan-code; the OUTPUT of kernel
venkatesh007c/floorplan-stage2-data (parser_{train,val}.zip, report/stats.npz).
If an earlier parser run's output is attached too, training resumes from its last.pt.
Output: /kaggle/working/runs/parser_v1/ (checkpoints/{last,best}.pt, train_log.csv,
val_log.csv, config.yaml, gpu_usage.csv).
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/parser"
RUN = "/kaggle/working/runs/parser_v1"


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
    from src.kaggle_kernel import find, gpu_count, gpu_monitor, run
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    for split in ("train", "val"):                       # extract to local disk (fast random access)
        z = find(f"parser_{split}.zip")[0]
        zipfile.ZipFile(z).extractall(f"{DATA}/{split}")
        print(split, len(os.listdir(f"{DATA}/{split}")) // 2, "plans from", z, flush=True)
    stats = find("report/stats.npz")[0]
    last = [p for p in find("runs/parser_v1/checkpoints/last.pt")]
    if last and not os.path.exists(f"{RUN}/checkpoints/last.pt"):
        shutil.copytree(os.path.dirname(os.path.dirname(last[0])), RUN, dirs_exist_ok=True)
        print("resuming from", last[0], flush=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # less memory fragmentation (OOM safeguard)
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    n = gpu_count()
    launcher = f"{sys.executable} -m torch.distributed.run --standalone --nproc_per_node={n}" if n > 1 else sys.executable
    try:
        run(f"{launcher} -m src.stage2.parse.train --config configs/stage2/parser_v1.yaml "
            f"--data {DATA} --out {RUN} --freq {stats}")
    finally:
        monitor.terminate()

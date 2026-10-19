"""Kaggle kernel floorplan-stage2-detv2 (GPU, one T4 used): train detector v2 (src/stage2/detect/train_detector_v2.py,
configs/stage2/detector_v2.yaml), warm-started from detector v1.

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (completion_train.zip),
floorplan-stage2-detv2-data (parsed_train.zip, parsed_val.zip), floorplan-stage2-k2a (runs/k2a_detector best.pt).
Output: /kaggle/working/runs/detector_v2/ (checkpoints, train_log.csv, val_log.csv, stdout.log, gpu_usage.csv).
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/detdata"
RUN = "/kaggle/working/runs/detector_v2"


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
    os.makedirs(DATA, exist_ok=True)
    for f in ("completion_train.zip", "parsed_train.zip", "parsed_val.zip"):
        shutil.copy(find(f)[0], f"{DATA}/{f}")
    init = find("runs/k2a_detector/checkpoints/best.pt")[0]
    os.makedirs(RUN, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    try:
        run(f"{sys.executable} -m src.stage2.detect.train_detector_v2 --config configs/stage2/detector_v2.yaml "
            f"--data {DATA} --parsed {DATA}/parsed_train.zip --val-parsed {DATA}/parsed_val.zip --init {init} --out {RUN}")
    finally:
        monitor.terminate()

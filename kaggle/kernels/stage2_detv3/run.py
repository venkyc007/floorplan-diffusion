"""Kaggle kernel floorplan-stage2-detv3 (GPU, one T4 used): detector v3 "direct" (drawing + parse -> missing type).

  1. crops (src/stage2/detect/make_crop_set.py): train drawings ~20 min, 4 crops per version; the TUNING val
     drawings, 2 crops per version (validation / comparison)
  2. training (src/stage2/detect/train_detector_v3.py, configs/stage2/detector_v3.yaml), ~1.9 h
  3. v1 / v2 / v3 on the same validation crops -> runs/detector_v3/compare.md

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (parser zips),
floorplan-stage2-parser-v2 (parser v2B), floorplan-stage2-k2a (detector v1), floorplan-stage2-detv2 (detector v2).
Output: /kaggle/working/runs/detector_v3/ (checkpoints, logs, compare.md, gpu_usage.csv).
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/parser"
CROPS = "/tmp/crops"
RUN = "/kaggle/working/runs/detector_v3"


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
    for split in ("train", "val"):
        zipfile.ZipFile(find(f"parser_{split}.zip")[0]).extractall(f"{DATA}/{split}")
    parser = find("runs/parser_v2B/checkpoints/best.pt")[0]
    v1 = find("runs/k2a_detector/checkpoints/best.pt")[0]
    v2 = find("runs/detector_v2/checkpoints/best.pt")[0]
    os.makedirs(CROPS, exist_ok=True)
    os.makedirs(RUN, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    monitor = gpu_monitor(f"{RUN}/gpu_usage.csv")
    try:
        run(f"{sys.executable} -m src.stage2.detect.make_crop_set --drawings {DATA}/train --parser {parser} "
            f"--out {CROPS}/crops_train.zip --minutes 20 --seed 0 --crops 4")
        run(f"{sys.executable} -m src.stage2.detect.make_crop_set --drawings {DATA}/val --parser {parser} "
            f"--out {CROPS}/crops_val.zip --ids configs/stage2/val_split/tuning_ids.txt --minutes 6 --seed 1 --crops 2")
        run(f"{sys.executable} -m src.stage2.detect.train_detector_v3 --config configs/stage2/detector_v3.yaml "
            f"--crops {CROPS}/crops_train.zip --val-crops {CROPS}/crops_val.zip --out {RUN} --v1 {v1} --v2 {v2}")
    finally:
        monitor.terminate()

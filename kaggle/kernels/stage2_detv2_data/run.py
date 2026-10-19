"""Kaggle kernel floorplan-stage2-detv2-data (GPU, one T4 used): detector v2 training data = parser v2B maps of
drawings with image-level omissions (src/stage2/detect/make_parsed_set.py).

  train: typed train drawings, 3 versions each (intact + 2 single omissions), ~38 min time budget
  val:   the TUNING val drawings (configs/stage2/val_split/tuning_ids.txt), same recipe, ~6 min (checkpoint choice)

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (parser zips) and
floorplan-stage2-parser-v2 (runs/parser_v2B/checkpoints/best.pt).
Output: /kaggle/working/parsed_train.zip, /kaggle/working/parsed_val.zip.
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/parser"
WORK = "/kaggle/working"


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
    from src.kaggle_kernel import find, run
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    for split in ("train", "val"):
        z = find(f"parser_{split}.zip")[0]
        zipfile.ZipFile(z).extractall(f"{DATA}/{split}")
        print(split, len(os.listdir(f"{DATA}/{split}")) // 2, "drawings", flush=True)
    parser = find("runs/parser_v2B/checkpoints/best.pt")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run(f"{sys.executable} -m src.stage2.detect.make_parsed_set --drawings {DATA}/train --parser {parser} "
        f"--out {WORK}/parsed_train.zip --minutes 38 --seed 0")
    run(f"{sys.executable} -m src.stage2.detect.make_parsed_set --drawings {DATA}/val --parser {parser} "
        f"--out {WORK}/parsed_val.zip --ids configs/stage2/val_split/tuning_ids.txt --minutes 6 --seed 1")

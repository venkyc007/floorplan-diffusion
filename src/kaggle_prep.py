"""Prepare folders for upload as private Kaggle datasets (see docs/kaggle_workflow.md).

    python -m src.kaggle_prep data   # -> data/kaggle_upload/floorplan-data/
    python -m src.kaggle_prep code   # -> data/kaggle_upload/floorplan-code/

data: train + val as compact label maps (N, 128, 128) uint8, compressed,
      plus metadata for train/val plans only. (Test data is not uploaded.)
code: src/, configs/ and requirements.txt zipped into code.zip.

Each folder gets the dataset-metadata.json the Kaggle CLI needs. Nothing is
uploaded here; uploading is a separate, deliberate `kaggle datasets` command.
"""

import argparse
import json
import zipfile

import numpy as np
import pandas as pd

from src.data.dataset import load_labels
from src.utils import PROJECT_ROOT, load_config, project_path

KAGGLE_USER = "venkatesh007c"
OUT = project_path("data/kaggle_upload")
DATASETS = {"data": "floorplan-data", "code": "floorplan-code"}


def write_metadata(folder, slug, title, license_name):
    meta = {"title": title, "id": f"{KAGGLE_USER}/{slug}", "licenses": [{"name": license_name}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def prepare_data():
    folder = OUT / DATASETS["data"]
    folder.mkdir(parents=True, exist_ok=True)
    proc = project_path(load_config("data")["paths"]["processed_dir"])
    for split in ("train", "val"):
        labels = load_labels(proc / f"{split}.npz")
        plan_ids = np.load(proc / f"{split}.npz")["plan_ids"]
        np.savez_compressed(folder / f"{split}.npz", labels=labels, plan_ids=plan_ids)
        print(f"{split}: {labels.shape} {labels.dtype}")
    meta = pd.read_csv(proc / "metadata.csv")
    meta[meta.split.isin(["train", "val"])].to_csv(folder / "metadata_trainval.csv", index=False)
    # Derived from ResPlan (CC BY 4.0): keep the same licence.
    write_metadata(folder, DATASETS["data"], "floorplan-data", "CC-BY-4.0")
    return folder


def prepare_code():
    folder = OUT / DATASETS["code"]
    folder.mkdir(parents=True, exist_ok=True)
    files = [p for d in ("src", "configs") for p in (PROJECT_ROOT / d).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    files.append(PROJECT_ROOT / "requirements.txt")
    with zipfile.ZipFile(folder / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(PROJECT_ROOT).as_posix())
    print(f"code.zip: {len(files)} files")
    write_metadata(folder, DATASETS["code"], "floorplan-code", "other")
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=["data", "code"])
    args = parser.parse_args()
    folder = prepare_data() if args.what == "data" else prepare_code()
    total = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
    for p in sorted(folder.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size / 1e6:8.2f} MB")
    print(f"total: {total / 1e6:.2f} MB in {folder}")


if __name__ == "__main__":
    main()

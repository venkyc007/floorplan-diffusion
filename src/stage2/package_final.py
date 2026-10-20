"""Build the private Kaggle dataset for the final Stage 2 test notebook (user, 2026-10-21). Laptop; uploads nothing.

    python -m src.stage2.package_final            -> data/kaggle_upload/floorplan-stage2-final/
    then (after asking):  cd data/kaggle_upload/floorplan-stage2-final && kaggle datasets create -p . -r zip

Contents (everything the notebook needs besides the code dataset and the test data):
  checkpoints/<path>   every checkpoint named in configs/stage2/pipeline.yaml (parser v2B, detectors v2 / v3,
                       16-class U-Net, constrained + control diffusion, B4 U-Net + diffusion, frozen Stage 1 U-Net +
                       diffusion), each at the same relative path the config uses (best.pt only)
  wheels/              pip wheels of segmentation-models-pytorch 0.3.4 (+ its pure-python dependencies) and
                       diffusers, so the notebook installs them offline (no internet needed)
  dataset-metadata.json  id venkatesh007c/floorplan-stage2-final, private
"""

import json
import shutil
import subprocess
import sys

import yaml

from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "kaggle_upload" / "floorplan-stage2-final"
WHEELS = ["segmentation-models-pytorch==0.3.4", "diffusers==0.30.3"]


def main():
    paths = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "pipeline.yaml").read_text(encoding="utf-8"))["checkpoints"]
    OUT.mkdir(parents=True, exist_ok=True)
    missing, total = [], 0
    for name, rel in paths.items():
        src = PROJECT_ROOT / rel
        if not src.exists():
            missing.append(f"{name}: {rel}")
            continue
        dst = OUT / "checkpoints" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        total += src.stat().st_size
        print(f"{name:18s} {src.stat().st_size / 1e6:7.1f} MB  {rel}")
    wheels = OUT / "wheels"
    wheels.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pip", "download", "--dest", str(wheels), "--only-binary=:all:",
                        "--platform", "manylinux2014_x86_64", "--python-version", "3.10", "--no-deps", *WHEELS,
                        "pretrainedmodels==0.7.4", "efficientnet-pytorch==0.7.1", "timm==0.9.2", "munch"],
                       capture_output=True, text=True)
    if r.returncode:                                    # sdists (pretrainedmodels / efficientnet) have no wheel
        subprocess.run([sys.executable, "-m", "pip", "download", "--dest", str(wheels), "--no-deps",
                        "pretrainedmodels==0.7.4", "efficientnet-pytorch==0.7.1"], check=False)
        subprocess.run([sys.executable, "-m", "pip", "download", "--dest", str(wheels), "--no-deps",
                        "--only-binary=:all:", *WHEELS, "timm==0.9.2", "munch"], check=False)
    (OUT / "dataset-metadata.json").write_text(json.dumps(
        {"title": "floorplan-stage2-final", "id": "venkatesh007c/floorplan-stage2-final",
         "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    print(f"checkpoints: {total / 1e6:.0f} MB; wheels: {sorted(p.name for p in wheels.iterdir())}")
    if missing:
        print("MISSING (download these kernel outputs first):\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()

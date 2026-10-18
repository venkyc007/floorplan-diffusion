"""Setup for the Stage 2 CPU evaluation kernels (kaggle/kernels/stage2_eval_*/run.py).

    from src.stage2.kaggle_cpu import setup, finish
    data = setup()          # packages, val zips into /tmp/s2data, checkpoints where configs/stage2/pipeline.yaml expects them
    run(f"python -m ... --zip {data}/parser_val.zip")
    finish()                # outputs/stage2/ files written by this run -> /kaggle/working/stage2/

Inputs (kernel_sources): floorplan-stage2-data (val zips), floorplan-stage2-parser-v1 (parser), floorplan-stage2-k2a
(detector + 16-class U-Net), floorplan-stage2-k1 (constrained diffusion). No GPU needed.
"""

import os
import shutil
import sys
import time

import yaml

from src.kaggle_kernel import CODE, WORK, find, run

DATA = "/tmp/s2data"
CKPTS = {"parser": "runs/parser_v1/checkpoints/best.pt", "detector": "runs/k2a_detector/checkpoints/best.pt",
         "unet": "runs/k2a_unet/checkpoints/best.pt", "diffusion": "runs/k1_constrained/checkpoints/best.pt"}
START = time.time()


def setup():
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.join(CODE, "outputs", "stage2"), exist_ok=True)   # outputs/ is not in the code bundle
    for f in ("completion_val.zip", "parser_val.zip"):
        shutil.copy(find(f)[0], f"{DATA}/{f}")
    with open(f"{CODE}/configs/stage2/pipeline.yaml", encoding="utf-8") as fh:
        paths = yaml.safe_load(fh)["checkpoints"]
    for name, pattern in CKPTS.items():
        src = find(pattern)
        if not src:
            raise SystemExit(f"checkpoint not found in the inputs: {pattern}")
        dst = os.path.join(CODE, paths[name])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src[0], dst)
        print(name, "<-", src[0], flush=True)
    return DATA


def finish():
    """Copy every outputs/stage2 file written or changed during this run to /kaggle/working/stage2/."""
    out = os.path.join(CODE, "outputs", "stage2")
    dst = os.path.join(WORK, "stage2")
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(out) if os.path.isdir(out) else []:
        p = os.path.join(out, f)
        if os.path.isfile(p) and os.path.getmtime(p) >= START:
            shutil.copy(p, dst)
            print("output:", f, flush=True)

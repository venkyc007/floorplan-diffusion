"""Kaggle kernel floorplan-novelty-data (GPU T4 x2): the large-omission set (src/novelty/build_set.py).

  train   every typed TRAIN drawing in random order: intact + 3 large omissions, parsed by the pipeline's PARSE;
          two shards, one per GPU, each stops after MINUTES_TRAIN -> large_train_0.zip, large_train_1.zip
  tune    the 212 TUNING val drawings: intact + 2 large omissions -> large_tune_0.zip, large_tune_1.zip
          (in-training validation and choosing thresholds; the HELD-OUT val drawings are not touched here)

Inputs: code dataset venkatesh007c/floorplan-code; outputs of kernels floorplan-stage2-data (parser_train.zip,
parser_val.zip) and floorplan-stage2-parser-v2 (parser v2B).
Output: /kaggle/working/large_set/*.zip (+ logs).
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
DATA = "/tmp/parser"
OUT = "/kaggle/working/large_set"
MINUTES_TRAIN = 50
MINUTES_TUNE = 12


def copy_code():
    hits = [h for h in glob.glob("/kaggle/input/**/src/utils.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)
    sys.path.insert(0, CODE)


def both_gpus(tag, cmd):
    """Run `cmd` (with {i} = shard / GPU index) once per GPU, in parallel; stop if either fails."""
    procs = []
    for i in range(2):
        log = open(f"{OUT}/{tag}_{i}.log", "w")
        procs.append(subprocess.Popen(cmd.format(i=i), shell=True, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "STAGE2_DEVICE": "cuda",
                                           "PYTHONUNBUFFERED": "1"}))
    codes = [p.wait() for p in procs]
    for i in range(2):
        print(f"===== tail of {tag}_{i}.log", flush=True)
        print("".join(open(f"{OUT}/{tag}_{i}.log").readlines()[-12:]), flush=True)
    if any(codes):
        raise SystemExit(f"{tag}: exit codes {codes}")


if __name__ == "__main__":
    copy_code()
    from src.kaggle_kernel import find, run
    run(f"{sys.executable} -m pip install -q segmentation-models-pytorch==0.3.4")
    try:
        import diffusers  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q diffusers")
    for split in ("train", "val"):
        zipfile.ZipFile(find(f"parser_{split}.zip")[0]).extractall(f"{DATA}/{split}")
    parser = find("runs/parser_v2B/checkpoints/best.pt")[0]
    os.makedirs(OUT, exist_ok=True)
    base = f"{sys.executable} -m src.novelty.build_set --parser {parser} --workers 2 --shard {{i}}/2 "
    both_gpus("tune", base + f"--drawings {DATA}/val --ids configs/stage2/val_split/tuning_ids.txt "
                             f"--out {OUT}/large_tune_{{i}}.zip --versions 2 --minutes {MINUTES_TUNE} --seed 1")
    both_gpus("train", base + f"--drawings {DATA}/train --out {OUT}/large_train_{{i}}.zip --versions 3 "
                              f"--minutes {MINUTES_TRAIN} --seed 0")
    run(f"ls -la {OUT}")

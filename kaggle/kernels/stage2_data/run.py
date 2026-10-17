"""Kaggle kernel (CPU): build the Stage 2 CubiCasa5K datasets from the SVGs.

Inputs: dataset qmarva/cubicasa5k (complete public mirror of Zenodo 2613548) and
our code dataset venkatesh007c/floorplan-code.
Output (/kaggle/working/stage2_data), attachable by later kernels via kernel_sources:
  parser_{train,val,test}.zip, completion_{train,val,test}.zip,
  report/{appliance_subtypes.yaml, metadata.csv, stats.npz, constraint_constants.yaml, data_stats.md,
          data_overlay.png}
Only report/ is downloaded to the laptop:
  kaggle kernels output venkatesh007c/floorplan-stage2-data -p outputs/kaggle/stage2_data --file-pattern "report/"
"""

import glob
import os
import shutil
import subprocess
import sys
import zipfile

CODE = "/tmp/code"
OUT = "/kaggle/working/stage2_data"


def copy_code():
    hits = [h for h in glob.glob("/kaggle/input/**/src/utils.py", recursive=True) if "floorplan-code" in h]
    if hits:
        shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE, dirs_exist_ok=True)
    else:
        z = [h for h in glob.glob("/kaggle/input/**/code.zip", recursive=True) if "floorplan-code" in h][0]
        zipfile.ZipFile(z).extractall(CODE)


def run(cmd):
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, cwd=CODE, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


if __name__ == "__main__":
    copy_code()
    root = os.path.dirname(sorted(glob.glob("/kaggle/input/**/train.txt", recursive=True))[0])
    print("CubiCasa root:", root, "| CPUs:", os.cpu_count(), flush=True)
    workers = max(1, os.cpu_count() or 1)
    # pass 1: which appliance sub-types are kitchen appliances (-> configs/stage2/appliance_subtypes.yaml)
    run(f"{sys.executable} -m src.stage2.data.survey --root {root} --workers {workers}")
    os.makedirs(f"{OUT}/report", exist_ok=True)
    shutil.copy(f"{CODE}/configs/stage2/appliance_subtypes.yaml", f"{OUT}/report/appliance_subtypes.yaml")
    # pass 2: the datasets and statistics
    run(f"{sys.executable} -m src.stage2.data.build_cubicasa --root {root} --out {OUT} --workers {workers}")
    run(f"{sys.executable} -m src.stage2.data.report --out {OUT} --root {root}")

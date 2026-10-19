"""Kaggle kernel floorplan-stage1-test-score (CPU): scores the saved Stage 1 TEST predictions of kernel
floorplan-stage1-test (python -m src.selection --benchmark test): per-case metrics, frozen size-adaptive hybrid,
paired bootstrap, tables by type and size.

Inputs: datasets floorplan-code + floorplan-test-benchmark; output of kernel floorplan-stage1-test (preds/*.npz).
Output: /kaggle/working/stage1_test.md (test only; the val-vs-test table is added locally) and
/kaggle/working/arch_eval_test/per_case/*.csv.gz.
"""

import glob
import os
import shutil
import sys
import zipfile

CODE = "/tmp/code"


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
    from src.kaggle_kernel import WORK, find, run
    bdir = f"{CODE}/data/benchmarks"
    os.makedirs(bdir, exist_ok=True)
    for f in ("test_benchmark.npz", "test_benchmark.csv"):
        shutil.copy(find(f)[0], f"{bdir}/{f}")
    preds = os.path.dirname(find("preds/hybrid_refine600.npz")[0])
    try:
        run(f"{sys.executable} -m src.selection --benchmark test --preds {preds} --out /tmp/out --workers 4")
    finally:
        if os.path.exists("/tmp/out/stage1_test.md"):
            shutil.copy("/tmp/out/stage1_test.md", WORK)
        if os.path.isdir("/tmp/out/arch_eval_test"):
            shutil.copytree("/tmp/out/arch_eval_test", f"{WORK}/arch_eval_test", dirs_exist_ok=True)

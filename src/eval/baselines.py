"""Simple "models" that set the floor (and ceiling) for the real model.

  do_nothing    fill the masked region with background
  nearest_fill  every masked pixel copies the class of the nearest known pixel
  ground_truth  the answer itself (must score perfectly: checks the metrics)

Run from the project folder (needs data/benchmarks/ from make_benchmark):
    python -m src.eval.baselines              # full test benchmark
    python -m src.eval.baselines --limit 300  # quick check

Saves outputs/baseline_results.md (+ .csv) and outputs/baseline_examples.png.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from src.data.classes import BACKGROUND
from src.eval.metrics import evaluate_benchmark, load_benchmark, to_markdown
from src.utils import project_path

# The columns shown in the summary table (all columns go to the CSV).
SHOW = ["n", "mIoU", "tmIoU", "IoU_wall", "tIoU_wall", "IoU_door", "tIoU_door",
        "IoU_window", "tIoU_window", "IoU_room", "IoU_outdoor", "pixel_acc",
        "F1_door", "F1_window", "F1_room", "conn_err", "open_on_wall",
        "room_leak", "room_count_err", "alignment"]


def do_nothing(batch):
    """Masked pixels become background; known pixels stay."""
    return np.where(batch["masks"], BACKGROUND, batch["corrupted"]).astype(np.uint8)


def nearest_fill(batch):
    """Each masked pixel takes the class of the nearest unmasked pixel."""
    out = []
    for corrupted, mask in zip(batch["corrupted"], batch["masks"]):
        # For every pixel: the coordinates of the nearest pixel where mask is False.
        _, (rows, cols) = distance_transform_edt(mask, return_indices=True)
        out.append(corrupted[rows, cols])
    return np.stack(out)


def ground_truth_fn(bench):
    """A 'model' that returns the true answer (looked up by position)."""
    state = {"i": 0}

    def predict(batch):
        n = len(batch["masks"])
        out = bench["targets"][state["i"]:state["i"] + n]
        state["i"] += n
        return out
    return predict


def subset(bench, meta, n):
    """First n samples of a benchmark (for quick checks)."""
    return {k: v[:n] for k, v in bench.items()}, meta.iloc[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples")
    args = parser.parse_args()
    bench, meta = load_benchmark(project_path("data/benchmarks/test_benchmark.npz"))
    if args.limit:
        bench, meta = subset(bench, meta, args.limit)

    tables = {}
    for name, fn in [("ground_truth", ground_truth_fn(bench)),
                     ("nearest_fill", nearest_fill), ("do_nothing", do_nothing)]:
        table, _ = evaluate_benchmark(fn, bench, meta)
        tables[name] = table
        print(f"{name}: done")

    out = project_path("outputs")
    overall = pd.DataFrame({name: t.loc["overall"] for name, t in tables.items()}).T
    overall.index.name = "baseline"
    pd.concat(tables, names=["baseline"]).to_csv(out / "baseline_results.csv")

    lines = [f"# Baseline results (test benchmark, {len(meta)} samples)", "",
             "## Overall", "", to_markdown(overall[SHOW]), ""]
    for name, t in tables.items():
        t.index.name = "group"
        lines += [f"## {name}: by type / location / position / size", "", to_markdown(t[SHOW]), ""]
    (out / "baseline_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(to_markdown(overall[SHOW]))


if __name__ == "__main__":
    main()

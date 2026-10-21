"""Figures for checking corruptions, the training dataset and baselines.

Run from the project folder (after make_benchmark):
    python -m src.stage1.eval.figures

Saves to outputs/:
  corruption_examples.png        per type, 3 examples: clean | model input | mask
  multi_corruption_examples.png  4 multi-part removals from the training dataset
  baseline_examples.png          6 test cases: input | do-nothing | nearest-fill | truth
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.stage1.data.classes import CLASSES, COLORS, NUM_CLASSES
from src.stage1.data.corruptions import TYPES, UNKNOWN
from src.stage1.data.dataset import CorruptionDataset
from src.stage1.eval.baselines import do_nothing, nearest_fill
from src.stage1.eval.metrics import load_benchmark
from src.utils import load_config, project_path

GREY = (150, 150, 150)  # how masked ("unknown") pixels are drawn


def colorize(label):
    """Label map -> RGB; UNKNOWN pixels grey."""
    palette = np.zeros((256, 3), np.uint8)
    for c in range(NUM_CLASSES):
        palette[c] = COLORS[c]
    palette[UNKNOWN] = GREY
    return palette[label]


def show(ax, img, title):
    ax.imshow(img, interpolation="nearest", cmap="gray")  # cmap only affects masks
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def add_legend(fig, ncol=None):
    handles = [Patch(facecolor=np.array(COLORS[c]) / 255, edgecolor="grey", label=CLASSES[c])
               for c in range(NUM_CLASSES)]
    handles.append(Patch(facecolor=np.array(GREY) / 255, label="masked (unknown)"))
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles), frameon=False, fontsize=9)


def corruption_examples(bench, meta, path, per_type=3):
    fig, axes = plt.subplots(len(TYPES), 3 * per_type, figsize=(2.1 * 3 * per_type, 2.4 * len(TYPES) + 0.5))
    for r, ctype in enumerate(TYPES):
        rows = meta.index[meta.type == ctype][:per_type * 50:50]  # spread over plans
        for k, i in enumerate(rows):
            m = meta.loc[i]
            info = f"{m.location}, {m.position}, {m.size_bin}"
            show(axes[r, 3 * k], colorize(bench["targets"][i]), f"{ctype}: clean (plan {m.plan_id})")
            show(axes[r, 3 * k + 1], colorize(bench["inputs"][i]), f"model input\n{info}")
            show(axes[r, 3 * k + 2], bench["masks"][i], f"mask ({m.mask_area} px)")
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def multi_examples(path, n=4):
    """Draw training samples until n multi-part removals turn up."""
    cfg = load_config("stage1/corruption")
    ds = CorruptionDataset(project_path("data/processed/val.npz"), cfg)  # val: small, same code path
    torch.manual_seed(1)
    found = []
    while len(found) < n:
        i = int(ds.rng().integers(len(ds)))
        mask, corrupted, meta = ds.corrupt_once(ds.labels[i])
        if meta["type"] == "multi":
            found.append((ds.labels[i], corrupted, mask, meta))
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.4 * n + 0.5))
    for r, (clean, corrupted, mask, meta) in enumerate(found):
        show(axes[r, 0], colorize(clean), "clean")
        show(axes[r, 1], colorize(corrupted), f"input: {meta['subtypes']}")
        show(axes[r, 2], mask, f"mask: union of {meta['n_components']}")
    add_legend(fig, ncol=4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def baseline_examples(bench, meta, path, n=6):
    # One example per type plus one large mask, each from a different plan.
    picks = [meta.index[meta.type == t][7 + 97 * k] for k, t in enumerate(TYPES)]
    picks = (picks + [meta.index[meta.size_bin == "large"][501]])[:n]
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.4 * n + 0.5))
    for r, i in enumerate(picks):
        batch = {"corrupted": bench["inputs"][i:i + 1], "masks": bench["masks"][i:i + 1]}
        m = meta.loc[i]
        show(axes[r, 0], colorize(bench["inputs"][i]), f"input ({m.type}, {m.size_bin})")
        show(axes[r, 1], colorize(do_nothing(batch)[0]), "do nothing")
        show(axes[r, 2], colorize(nearest_fill(batch)[0]), "nearest fill")
        show(axes[r, 3], colorize(bench["targets"][i]), "ground truth")
    add_legend(fig, ncol=4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    out = project_path("outputs/stage1")
    bench, meta = load_benchmark(project_path("data/benchmarks/test_benchmark.npz"))
    corruption_examples(bench, meta, out / "corruption_examples.png")
    multi_examples(out / "multi_corruption_examples.png")
    baseline_examples(bench, meta, out / "baseline_examples.png")
    print("figures saved to", out)

"""Sanity checks on the processed dataset. Saves pictures to outputs/.

Run from the project folder (after build_dataset):
    python -m src.data.checks

  1. opening_sizes.png   How big are doors and windows, in pixels?
                         (Tiny openings are hard for a model and for metrics.)
  2. duplicate_pairs.png 5 removed near-duplicates next to the plan kept
                         instead, to see that removal was justified.
"""

import random

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.classes import DOOR, WINDOW
from src.data.rasterize import rasterize_plan
from src.data.resplan import load_plans
from src.utils import load_config, project_path
from src.visualize import save_grid

SMALL = 6  # openings below this many pixels are counted as "small"


def component_sizes(masks, cls):
    """Pixel count of every separate door (or window) in every plan.

    A "component" = a group of touching pixels of that class
    (4-connectivity: touching by an edge, not just a corner).
    """
    sizes = []
    for m in masks:
        n, _, stats, _ = cv2.connectedComponentsWithStats(m[cls], connectivity=4)
        sizes.extend(stats[1:, cv2.CC_STAT_AREA])  # row 0 is the background
    return np.array(sizes)


def opening_sizes(cfg, out_dir):
    proc = project_path(cfg["paths"]["processed_dir"])
    masks = np.concatenate([np.load(proc / f"{s}.npz")["masks"] for s in ("train", "val", "test")])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cls, name in zip(axes, (DOOR, WINDOW), ("door", "window")):
        sizes = component_sizes(masks, cls)
        small = (sizes < SMALL).mean()
        print(f"{name}s: {len(sizes)} components | min {sizes.min()} px | "
              f"median {np.median(sizes):.0f} px | under {SMALL} px: {small:.2%}")
        ax.hist(sizes, bins=np.arange(0, np.percentile(sizes, 99.5) + 2), color="grey")
        ax.axvline(SMALL, color="red", ls="--", label=f"{SMALL} px ({small:.1%} below)")
        ax.set(title=f"{name} sizes ({len(sizes)} components, {len(masks)} plans)",
               xlabel="pixels per component", ylabel="count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "opening_sizes.png", dpi=110)
    plt.close(fig)


def duplicate_pairs(cfg, out_dir, n=5):
    proc = project_path(cfg["paths"]["processed_dir"])
    meta = pd.read_csv(proc / "metadata.csv")
    pairs = pd.read_csv(proc / "similar_pairs.csv")
    removed = meta[meta.split == "removed_duplicate"]
    pick = removed.sample(n, random_state=1)

    by_id = {p["id"]: p for p in load_plans(project_path(cfg["paths"]["pickle"]))}
    size, padding = cfg["raster"]["size"], cfg["raster"]["padding"]
    labels, titles = [], []
    for rid, kid in zip(pick.plan_id, pick.duplicate_of):
        # Similarity of this exact pair (if they were linked only through a
        # chain of other plans, there is no direct pair -> show "chain").
        hit = pairs[((pairs.plan_id_a == rid) & (pairs.plan_id_b == kid)) |
                    ((pairs.plan_id_a == kid) & (pairs.plan_id_b == rid))]
        sim = f"{hit.similarity.iloc[0]:.3f}" if len(hit) else "chain"
        labels += [rasterize_plan(by_id[rid], size, padding)[0],
                   rasterize_plan(by_id[kid], size, padding)[0]]
        titles += [f"REMOVED {rid} (sim {sim})", f"kept {kid}"]
    save_grid(labels, titles, out_dir / "duplicate_pairs.png", ncols=2)
    print(f"Near-duplicate removal threshold: similarity >= {cfg['dedup']['remove_threshold']}")


if __name__ == "__main__":
    cfg = load_config("data")
    out_dir = project_path("outputs")
    opening_sizes(cfg, out_dir)
    duplicate_pairs(cfg, out_dir)

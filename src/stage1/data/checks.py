"""Sanity checks on the processed dataset. Saves pictures to outputs/.

Run from the project folder (after build_dataset):
    python -m src.stage1.data.checks

  1. opening_sizes.png    How big are doors and windows, in pixels?
  2. duplicate_pairs.png  8 removed near-duplicates next to the plan kept
                          instead, with their IoU score.
  3. anomalies.png        The 16 plans whose outdoor area looks most suspicious.
  4. raw_vs_processed.png Raw ResPlan polygons next to our 6-channel image,
                          to check both use the same coordinate frame.
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as PolygonPatch

from src.stage1.data.classes import (COLORS, DOOR, OUTDOOR, RESPLAN_FIELDS, ROOM,
                              WALL, WINDOW)
from src.stage1.data.resplan import get_polygons, load_plans
from src.utils import load_config, project_path
from src.stage1.visualize import colorize, save_grid

# Plans shown in raw_vs_processed.png: two augmented plans (ResPlan bug:
# inner/land not rotated), a plan without land, a plan with an oversized
# land shape, and a plan whose 'room' polygon covers the whole plot.
RAW_CHECK_IDS = [485548, 680096, 15861, 12117, 2093]
SMALL = 6        # openings below this many pixels are counted as "small"
FAR_PX = 10      # outdoor pixels further than this from any wall are "far"
SPLITS = ("train", "val", "test")


def load_all(cfg):
    """Label maps of ALL plans + metadata, in the same order."""
    proc = project_path(cfg["paths"]["processed_dir"])
    data = np.load(proc / "all_labels.npz")
    meta = pd.read_csv(proc / "metadata.csv")
    assert (data["plan_ids"] == meta.plan_id.to_numpy()).all()
    return data["labels"], meta


# ------------------------------------------------------------ 1. openings

def component_sizes(labels, cls):
    """Pixel count of every separate door (or window) in every plan.

    A "component" = a group of touching pixels of that class
    (4-connectivity: touching by an edge, not just a corner).
    """
    sizes = []
    for label in labels:
        mask = (label == cls).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        sizes.extend(stats[1:, cv2.CC_STAT_AREA])  # row 0 is the background
    return np.array(sizes)


def opening_sizes(labels, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cls, name in zip(axes, (DOOR, WINDOW), ("door", "window")):
        sizes = component_sizes(labels, cls)
        small = (sizes < SMALL).mean()
        print(f"{name}s: {len(sizes)} components | min {sizes.min()} px | "
              f"median {np.median(sizes):.0f} px | under {SMALL} px: {small:.2%}")
        ax.hist(sizes, bins=np.arange(0, np.percentile(sizes, 99.5) + 2), color="grey")
        ax.axvline(SMALL, color="red", ls="--", label=f"{SMALL} px ({small:.1%} below)")
        ax.set(title=f"{name} sizes ({len(sizes)} components, {len(labels)} plans)",
               xlabel="pixels per component", ylabel="count")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "opening_sizes.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------ 2. duplicate pairs

def duplicate_pairs(labels, meta, out_dir, n=8, seed=1):
    row_of = {pid: r for r, pid in enumerate(meta.plan_id)}
    pick = meta[meta.split == "removed_duplicate"].sample(n, random_state=seed)
    images, titles = [], []
    for rid, kid, iou in zip(pick.plan_id, pick.duplicate_of, pick.duplicate_iou):
        images += [labels[row_of[rid]], labels[row_of[kid]]]
        titles += [f"REMOVED {rid} (tol. IoU {iou:.3f})", f"kept {kid} ({meta.split[row_of[kid]]})"]
    save_grid(images, titles, out_dir / "duplicate_pairs.png", ncols=4)


# ------------------------------------------------------------ 3. anomalies

def bbox(mask):
    """(row0, col0, row1, col1) of the True pixels, or None if there are none."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return rows.min(), cols.min(), rows.max() + 1, cols.max() + 1


def box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def anomaly_metrics(label):
    """Numbers describing how plausible a plan's outdoor area is."""
    indoor = np.isin(label, [ROOM, WALL, DOOR, WINDOW])
    outdoor = label == OUTDOOR
    boundary = np.isin(label, [WALL, DOOR, WINDOW])
    # Distance (pixels) from every pixel to the nearest wall/door/window pixel.
    dist = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    bi, bo = bbox(indoor), bbox(outdoor)
    if bo is None:
        overlap = np.nan
    else:
        inter = (max(bi[0], bo[0]), max(bi[1], bo[1]), min(bi[2], bo[2]), min(bi[3], bo[3]))
        overlap = box_area(inter) / box_area(bo)  # share of the outdoor box inside the indoor box
    return {
        "outdoor_ratio": outdoor.sum() / indoor.sum(),
        "far_outdoor_px": int((outdoor & (dist > FAR_PX)).sum()),
        "bbox_overlap": overlap,
    }


def anomaly_scan(labels, meta, out_dir, title=""):
    """Count suspicious plans and save the 16 worst to anomalies.png."""
    m = pd.DataFrame([anomaly_metrics(l) for l in labels])
    flags = pd.DataFrame({
        "outdoor > 1.5x indoor": m.outdoor_ratio > 1.5,
        f"outdoor pixels > {FAR_PX} px from any wall": m.far_outdoor_px > 0,
        "outdoor box barely overlaps indoor box (<10%)": m.bbox_overlap < 0.10,
    })
    groups = {"all": np.ones(len(meta), bool), "augmented": meta.is_augmented.to_numpy(),
              "not augmented": ~meta.is_augmented.to_numpy()}
    print(f"Anomaly scan {title}")
    for name, flag in flags.items():
        counts = ", ".join(f"{g}: {int(flag[sel].sum())}/{int(sel.sum())}" for g, sel in groups.items())
        print(f"  {name}: {counts}")

    # Rank: most "far" outdoor pixels first, then by outdoor/indoor ratio.
    worst = m.sort_values(["far_outdoor_px", "outdoor_ratio"], ascending=False).index[:16]
    save_grid([labels[i] for i in worst],
              [f"{meta.plan_id[i]}{' AUG' if meta.is_augmented[i] else ''} far={m.far_outdoor_px[i]} "
               f"r={m.outdoor_ratio[i]:.1f}" for i in worst],
              out_dir / "anomalies.png")
    return m


# ----------------------------------------------------- 4. raw vs processed

def draw_raw(ax, plan):
    """Draw a plan's polygons exactly as stored in ResPlan.pkl (no conversion).

    'inner' (the indoor outline) is drawn as a dashed line: it should sit
    exactly inside the walls if all fields share one coordinate frame.
    """
    for cls in (OUTDOOR, ROOM, WALL, WINDOW, DOOR):
        color = np.array(COLORS[cls]) / 255
        for field in RESPLAN_FIELDS[cls]:
            for poly in get_polygons(plan, field):
                ax.add_patch(PolygonPatch(np.array(poly.exterior.coords), fc=color, ec="none"))
    for poly in get_polygons(plan, "inner"):
        ax.add_patch(PolygonPatch(np.array(poly.exterior.coords), fill=False,
                                  ec="purple", ls="--", lw=1.2))
    ax.autoscale()
    ax.set_aspect("equal")


def raw_vs_processed(cfg, labels, meta, plan_ids, out_dir):
    by_id = {p["id"]: p for p in load_plans(project_path(cfg["paths"]["pickle"]))}
    row_of = {pid: r for r, pid in enumerate(meta.plan_id)}
    fig, axes = plt.subplots(len(plan_ids), 2, figsize=(8, 4 * len(plan_ids)))
    for (ax_raw, ax_img), pid in zip(axes, plan_ids):
        aug = " (augmented)" if meta.is_augmented[row_of[pid]] else ""
        draw_raw(ax_raw, by_id[pid])
        ax_raw.set_title(f"{pid}{aug}: raw polygons\n(dashed purple = 'inner' outline)", fontsize=9)
        ax_img.imshow(colorize(labels[row_of[pid]]), interpolation="nearest")
        ax_img.set_title(f"{pid}: our 128x128 image", fontsize=9)
        ax_img.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "raw_vs_processed.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config("stage1/data")
    out_dir = project_path("outputs/stage1")
    labels, meta = load_all(cfg)
    in_split = meta.split.isin(SPLITS).to_numpy()
    opening_sizes(labels[in_split], out_dir)
    duplicate_pairs(labels, meta, out_dir)
    anomaly_scan(labels[in_split], meta[in_split].reset_index(drop=True), out_dir,
                 title=f"(final dataset: {in_split.sum()} plans)")
    raw_vs_processed(cfg, labels, meta, RAW_CHECK_IDS, out_dir)
    print(f"Near-duplicate removal: tolerant mean per-class IoU >= {cfg['dedup']['remove_iou']}"
          f" ({cfg['dedup']['iou_tolerance']} px tolerance)")

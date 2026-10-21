"""Build the processed dataset: the main data-pipeline script.

Run from the project folder:
    python -m src.stage1.data.build_dataset --limit 100   # quick test on 100 plans
    python -m src.stage1.data.build_dataset               # all ~17,000 plans
    python -m src.stage1.data.build_dataset --reuse-split # re-convert, but keep the
                                                   # existing duplicate removal
                                                   # and train/val/test split

What it does:
  1. Load ResPlan and convert every plan to a 128x128 label map.
  2. Leave out ResPlan's "augmented" plans (broken copies, see configs/stage1/data.yaml).
  3. Find near-duplicates; remove each plan that directly matches a kept plan.
  4. Split the remaining plans 80/10/10, keeping similar plans together.
     (Steps 2-4 are read from the old metadata.csv when --reuse-split is used.)
  5. Save to data/processed/:
       train.npz, val.npz, test.npz   arrays of shape (N, 6, 128, 128) + plan ids
       all_labels.npz                 label maps (N, 128, 128) of ALL plans,
                                      incl. removed ones (for checks/figures)
       metadata.csv                   one row per plan (including removed ones)
       similar_pairs.csv              every candidate pair with its scores
  6. Save a preview of 16 random plans to outputs/stage1/samples.png.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from shapely.validation import make_valid

from src.stage1.data.classes import RESPLAN_FIELDS, DOOR, ROOM, WINDOW
from src.stage1.data.dedup import (confirm_candidates, find_candidates, group_pairs,
                            remove_duplicates)
from src.stage1.data.rasterize import rasterize_plan, to_onehot
from src.stage1.data.resplan import get_polygons, load_plans
from src.stage1.data.split import split_groups
from src.utils import load_config, project_path
from src.stage1.visualize import save_grid

# Plans whose implied wall thickness (see real_scale) falls outside this
# range (metres) probably have a wrong 'area' value -> flagged scale_ok=False.
PLAUSIBLE_WALL_M = (0.10, 0.40)


def count_parts(plan, fields):
    """How many separate shapes (rooms, doors, ...) a plan has in these fields."""
    return sum(len(get_polygons(plan, f)) for f in fields)


def total_area(plan, field):
    """Area of one field in plan units^2 (make_valid repairs broken shapes)."""
    return sum(make_valid(p).area for p in get_polygons(plan, field))


def real_scale(plan, px_per_unit):
    """Work out how big one pixel is in real life.

    ResPlan coordinates are not metres, but 'area' (m^2, from the listing)
    matches the indoor area + walls. So:
        metres per unit  = sqrt(area_m2 / (indoor + wall area in units^2))
        metres per pixel = metres per unit / pixels per unit
    Check: this gives wall thicknesses of 0.16-0.33 m for 98% of plans,
    matching the ResPlan README. Plans outside 0.10-0.40 m get scale_ok=False.
    """
    area_m2 = float(plan["area"])
    unit_area = total_area(plan, "inner") + total_area(plan, "wall")
    metres_per_unit = np.sqrt(area_m2 / unit_area) if area_m2 > 0 and unit_area > 0 else np.nan
    wall_m = plan["wall_depth"] * metres_per_unit
    return {
        "metres_per_pixel": metres_per_unit / px_per_unit,
        "wall_thickness_m": wall_m,
        "scale_ok": bool(PLAUSIBLE_WALL_M[0] <= wall_m <= PLAUSIBLE_WALL_M[1]),
    }


def describe_plan(plan, scale, augmented_ids, resplan_split):
    """One metadata row (a dictionary) for one plan."""
    return {
        "plan_id": plan["id"],
        "area_m2": float(plan["area"]),           # real area (from the listing)
        "net_area_m2": float(plan["net_area"]),
        "n_rooms": count_parts(plan, RESPLAN_FIELDS[ROOM]),
        "n_bedrooms": count_parts(plan, ["bedroom"]),
        "n_bathrooms": count_parts(plan, ["bathroom"]),
        "n_balconies": count_parts(plan, ["balcony"]),
        "n_doors": count_parts(plan, RESPLAN_FIELDS[DOOR]),
        "n_windows": count_parts(plan, RESPLAN_FIELDS[WINDOW]),
        "wall_depth": float(plan["wall_depth"]),  # in plan units
        "px_per_unit": scale,                     # plan units -> pixels
        **real_scale(plan, scale),                # pixels -> metres
        "is_augmented": plan["id"] in augmented_ids,
        "resplan_split": resplan_split.get(plan["id"], ""),
    }


def rasterize_all(plans, cfg):
    """Step 1: label maps (N, 128, 128) + a metadata table, one row per plan."""
    official = json.load(open(project_path(cfg["paths"]["resplan_split"])))
    augmented_ids = set(official["augmented"])
    resplan_split = {pid: name for name, ids in official.items() for pid in ids}

    size, padding = cfg["raster"]["size"], cfg["raster"]["padding"]
    labels, rows = [], []
    for i, plan in enumerate(plans):
        label, scale = rasterize_plan(plan, size, padding)
        labels.append(label)
        rows.append(describe_plan(plan, scale, augmented_ids, resplan_split))
        if (i + 1) % 2000 == 0:
            print(f"  rasterized {i + 1}/{len(plans)}")
    return np.stack(labels), pd.DataFrame(rows)


def dedup_and_split(labels, df, cfg, out_dir):
    """Steps 2-4: add 'split', 'duplicate_of', 'duplicate_iou' and
    'similarity_group' columns. 'split' is train / val / test, or
    excluded_augmented / removed_duplicate for plans left out."""
    d = cfg["dedup"]
    df["split"] = ""
    df["duplicate_of"] = -1
    df["duplicate_iou"] = np.nan

    # ---- 2. Leave out ResPlan's augmented copies
    if cfg["exclude_augmented"]:
        df.loc[df.is_augmented, "split"] = "excluded_augmented"
    pool = np.flatnonzero(df.split == "")          # rows still in play
    print(f"Excluded {len(df) - len(pool)} augmented plans; {len(pool)} remain")

    # ---- 3. Near-duplicates (positions below are positions within `pool`)
    candidates = find_candidates(labels[pool], d["grid"], d["candidate_cosine"])
    print(f"Stage 1: {len(candidates)} candidate pairs (cosine >= {d['candidate_cosine']})")
    pairs = confirm_candidates(labels[pool], candidates, d["iou_tolerance"])
    ids = df.plan_id.to_numpy()[pool]
    pd.DataFrame(
        [(ids[i], ids[j], round(cos, 4), round(iou, 4)) for i, j, cos, iou in pairs],
        columns=["plan_id_a", "plan_id_b", "cosine", "iou"],
    ).to_csv(out_dir / "similar_pairs.csv", index=False)

    priority = np.argsort(ids, kind="stable")      # keep the lowest plan id first
    dup_of, dup_iou = remove_duplicates(len(pool), pairs, d["remove_iou"], priority)
    removed = dup_of >= 0
    df.loc[pool[removed], "split"] = "removed_duplicate"
    df.loc[pool[removed], "duplicate_of"] = ids[dup_of[removed]]
    df.loc[pool[removed], "duplicate_iou"] = dup_iou[removed].round(4)
    print(f"Stage 2: removed {removed.sum()} near-duplicates (tolerant IoU >= {d['remove_iou']})")

    # ---- 4. Split the kept plans, in groups of similar plans
    kept = np.flatnonzero(~removed)                # positions within pool
    position = {p: k for k, p in enumerate(kept)}  # pool position -> kept position
    links = [(position[i], position[j]) for i, j, cos, _ in pairs
             if cos >= d["group_cosine"] and i in position and j in position]
    groups = group_pairs(len(kept), links)
    s = cfg["split"]
    df.loc[pool[kept], "split"] = split_groups(
        groups, {k: s[k] for k in ("train", "val", "test")}, s["seed"])
    df["similarity_group"] = -1
    df.loc[pool[kept], "similarity_group"] = ids[kept][groups]  # name group by a member id
    return df


def reuse_split(df, out_dir):
    """Steps 2-4 alternative: copy the columns from the existing metadata.csv."""
    old = pd.read_csv(out_dir / "metadata.csv").set_index("plan_id")
    missing = set(df.plan_id) - set(old.index)
    assert not missing, f"{len(missing)} plans are not in the old metadata.csv"
    for col in ("split", "duplicate_of", "duplicate_iou", "similarity_group"):
        df[col] = df.plan_id.map(old[col]).to_numpy()
    print("Reused augmented exclusion + duplicate removal + split from metadata.csv")
    return df


def check_no_leakage(df, cfg, out_dir):
    """Checks that (a) no similar pair of kept plans crosses splits, and
    (b) every removed plan directly matches a KEPT plan (no chains)."""
    d = cfg["dedup"]
    pairs = pd.read_csv(out_dir / "similar_pairs.csv")
    split_of = df.set_index("plan_id")["split"]
    in_split = ("train", "val", "test")

    linked = pairs[pairs.cosine >= d["group_cosine"]]
    a, b = linked.plan_id_a.map(split_of), linked.plan_id_b.map(split_of)
    crossing = int((a.isin(in_split) & b.isin(in_split) & (a != b)).sum())
    assert crossing == 0, f"{crossing} similar pairs cross splits!"

    removed = df[df.split == "removed_duplicate"]
    assert removed.duplicate_of.map(split_of).isin(in_split).all(), "removed plan points to a non-kept plan"
    assert (removed.duplicate_iou >= d["remove_iou"]).all(), "removed plan below the IoU threshold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N plans (for testing)")
    parser.add_argument("--reuse-split", action="store_true",
                        help="keep the exclusion + duplicate removal + split in the existing metadata.csv")
    args = parser.parse_args()
    cfg = load_config("stage1/data")
    out_dir = project_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and rasterize ------------------------------------------
    t0 = time.time()
    plans = load_plans(project_path(cfg["paths"]["pickle"]))
    if args.limit:
        plans = plans[:args.limit]
    print(f"Loaded {len(plans)} plans ({time.time() - t0:.0f}s)")
    labels, df = rasterize_all(plans, cfg)
    print(f"Rasterized ({time.time() - t0:.0f}s)")
    np.savez_compressed(out_dir / "all_labels.npz", labels=labels,
                        plan_ids=df.plan_id.to_numpy())

    # ---- 2-4. Exclusion + near-duplicates + split --------------------------
    if args.reuse_split:
        df = reuse_split(df, out_dir)
    else:
        df = dedup_and_split(labels, df, cfg, out_dir)
    check_no_leakage(df, cfg, out_dir)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Plans with a plausible real scale: {df.scale_ok.sum()}/{len(df)}")

    # ---- 5. Save arrays ------------------------------------------------
    split_arr = df["split"].to_numpy()
    for name in ("train", "val", "test"):
        rows_in_split = np.flatnonzero(split_arr == name)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            masks=np.stack([to_onehot(l) for l in labels[rows_in_split]]),
            plan_ids=df["plan_id"].to_numpy()[rows_in_split],
        )
        print(f"  {name}: {len(rows_in_split)} plans")

    # ---- 6. Preview -------------------------------------------------------
    kept = np.flatnonzero(np.isin(split_arr, ["train", "val", "test"]))
    pick = random.Random(0).sample(list(kept), min(16, len(kept)))
    save_grid([labels[i] for i in pick],
              [f"id {df.plan_id[i]} ({df.split[i]})" for i in pick],
              project_path(cfg["paths"]["samples_png"]))
    print(f"Done in {time.time() - t0:.0f}s. Files in {out_dir}")


if __name__ == "__main__":
    main()

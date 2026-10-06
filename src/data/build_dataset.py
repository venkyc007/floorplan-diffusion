"""Build the processed dataset: the main data-pipeline script.

Run from the project folder:
    python -m src.data.build_dataset --limit 100   # quick test on 100 plans
    python -m src.data.build_dataset               # all ~17,000 plans
    python -m src.data.build_dataset --reuse-split # re-convert, but keep the
                                                   # existing duplicate removal
                                                   # and train/val/test split

What it does:
  1. Load ResPlan and convert every plan to a 128x128 label map.
  2. Find near-duplicates and remove all but one copy of each.
  3. Split the remaining plans 80/10/10, keeping similar plans together.
     (Steps 2-3 are read from the old metadata.csv when --reuse-split is used.)
  4. Save to data/processed/:
       train.npz, val.npz, test.npz   arrays of shape (N, 6, 128, 128) + plan ids
       metadata.csv                   one row per plan (including removed ones)
       similar_pairs.csv              every similar pair found, for checking
  5. Save a preview of 16 random plans to outputs/samples.png.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from shapely.validation import make_valid

from src.data.classes import RESPLAN_FIELDS, DOOR, ROOM, WINDOW
from src.data.dedup import find_similar_pairs, group_pairs
from src.data.rasterize import rasterize_plan, to_onehot
from src.data.resplan import get_polygons, load_plans
from src.data.split import split_groups
from src.utils import load_config, project_path
from src.visualize import save_grid

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


def choose_keepers(df, dup_group):
    """In each duplicate group keep ONE plan: an original (non-augmented)
    one if possible, then the lowest plan id. Returns the row index of the
    kept plan for every row (a row that is kept points to itself)."""
    best_first = df.assign(dup_group=dup_group).sort_values(["is_augmented", "plan_id"])
    first_in_group = best_first.groupby("dup_group").head(1)
    keeper_row = pd.Series(first_in_group.index, index=first_in_group["dup_group"])
    return keeper_row.loc[dup_group].to_numpy()


def rasterize_all(plans, cfg):
    """Steps 1: label maps (N, 128, 128) + a metadata table, one row per plan."""
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
    """Steps 2-3: add 'duplicate_of', 'split' and 'similarity_group' columns."""
    d = cfg["dedup"]
    pairs = find_similar_pairs(labels, d["grid"], min(d["remove_threshold"], d["group_threshold"]))
    keeper = choose_keepers(df, group_pairs(len(df), pairs, d["remove_threshold"]))
    removed = keeper != np.arange(len(df))
    df["duplicate_of"] = np.where(removed, df["plan_id"].to_numpy()[keeper], -1)
    print(f"Found {len(pairs)} similar pairs; removing {removed.sum()} near-duplicates "
          f"({removed[df['is_augmented']].sum()} of them augmented copies)")

    pd.DataFrame(
        [(df.plan_id[i], df.plan_id[j], round(s, 4)) for i, j, s in pairs],
        columns=["plan_id_a", "plan_id_b", "similarity"],
    ).to_csv(out_dir / "similar_pairs.csv", index=False)

    # Split only the kept plans, in groups of similar plans.
    kept = np.flatnonzero(~removed)
    position = {row: k for k, row in enumerate(kept)}  # row in df -> position in kept
    kept_pairs = [(position[i], position[j], s) for i, j, s in pairs
                  if i in position and j in position]
    groups = group_pairs(len(kept), kept_pairs, d["group_threshold"])
    s = cfg["split"]
    splits = split_groups(groups, {k: s[k] for k in ("train", "val", "test")}, s["seed"])

    df["split"] = "removed_duplicate"
    df.loc[kept, "split"] = splits
    df["similarity_group"] = -1
    df.loc[kept, "similarity_group"] = df.plan_id.to_numpy()[kept][groups]  # name group by a member id
    return df


def reuse_split(df, out_dir):
    """Steps 2-3 alternative: copy the columns from the existing metadata.csv."""
    old = pd.read_csv(out_dir / "metadata.csv").set_index("plan_id")
    missing = set(df.plan_id) - set(old.index)
    assert not missing, f"{len(missing)} plans are not in the old metadata.csv"
    for col in ("duplicate_of", "split", "similarity_group"):
        df[col] = df.plan_id.map(old[col]).to_numpy()
    print("Reused duplicate removal + split from the existing metadata.csv")
    return df


def check_no_leakage(df, cfg, out_dir):
    """No similar pair (above the group threshold) may cross splits."""
    pairs = pd.read_csv(out_dir / "similar_pairs.csv")
    pairs = pairs[pairs.similarity >= cfg["dedup"]["group_threshold"]]
    split_of = df.set_index("plan_id")["split"]
    a, b = pairs.plan_id_a.map(split_of), pairs.plan_id_b.map(split_of)
    both_kept = (a != "removed_duplicate") & (b != "removed_duplicate")
    crossing = int((both_kept & (a != b)).sum())
    assert crossing == 0, f"{crossing} similar pairs cross splits!"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N plans (for testing)")
    parser.add_argument("--reuse-split", action="store_true",
                        help="keep the duplicate removal + split in the existing metadata.csv")
    args = parser.parse_args()
    cfg = load_config("data")
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

    # ---- 2-3. Near-duplicates + split -----------------------------------
    if args.reuse_split:
        df = reuse_split(df, out_dir)
    else:
        df = dedup_and_split(labels, df, cfg, out_dir)
    check_no_leakage(df, cfg, out_dir)
    df.to_csv(out_dir / "metadata.csv", index=False)
    print(f"Plans with a plausible real scale: {df.scale_ok.sum()}/{len(df)}")

    # ---- 4. Save arrays ------------------------------------------------
    split_arr = df["split"].to_numpy()
    for name in ("train", "val", "test"):
        rows_in_split = np.flatnonzero(split_arr == name)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            masks=np.stack([to_onehot(l) for l in labels[rows_in_split]]),
            plan_ids=df["plan_id"].to_numpy()[rows_in_split],
        )
        print(f"  {name}: {len(rows_in_split)} plans")

    # ---- 5. Preview -------------------------------------------------------
    kept = np.flatnonzero(split_arr != "removed_duplicate")
    pick = random.Random(0).sample(list(kept), min(16, len(kept)))
    save_grid([labels[i] for i in pick],
              [f"id {df.plan_id[i]} ({df.split[i]})" for i in pick],
              project_path(cfg["paths"]["samples_png"]))
    print(f"Done in {time.time() - t0:.0f}s. Files in {out_dir}")


if __name__ == "__main__":
    main()

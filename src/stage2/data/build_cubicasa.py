"""Build the Stage 2 datasets from CubiCasa5K's SVGs (runs on Kaggle, CPU only).

    python -m src.stage2.data.build_cubicasa --root <folder with train.txt> --out <dir>
        [--limit N] [--workers 4]

For every plan in the official train/val/test lists:
  1. parse model.svg (cubicasa_svg.py) and measure the median wall thickness t
     from the wall polygons;
  2. PARSER set: scale s = 8 / t (long side capped at 2048 px): greyscale drawing
     (from the higher-resolution of F1_original / F1_scaled) + label map;
  3. COMPLETION set: scale s = 3 / t: label map only;
  4. measure statistics on the completion map (plan_stats.py).
Both label maps are drawn straight from the polygons, never downsampled.

Output (<out>):
  parser_{train,val,test}.zip      <plan>_img.png (greyscale) + <plan>_lbl.png (uint8 class ids)
  completion_{train,val,test}.zip  <plan>.png (uint8 class ids); untyped plans left out
Untyped plans (raw "Undefined" > 50% of room pixels) stay in the parser set with their
room-type pixels set to 255 (ignore) and are left out of the completion set and all statistics.
Needs configs/stage2/appliance_subtypes.yaml from the survey pass (survey.py).
  report/metadata.csv              one row per plan (kept or dropped, with the reason)
  report/stats.npz                 pooled statistics for report.py
Plan id = "<category>-<number>", e.g. "high_quality_architectural-6044".
"""

import argparse
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.stage2.data import plan_stats as ps
from src.stage2.data.classes16 import C16
from src.stage2.data.cubicasa_svg import opening_length, parse_svg, thickness_length, wall_thickness
from src.stage2.data.rasterise import draw_labels, fixture_footprint, load_grey, make_image, untyped_rooms, whiten_padding
from src.utils import load_config

CFG = load_config("stage2/data")
IGNORE = 255   # parser label value for "no label" (room types of untyped plans)


def find_root(path):
    hits = sorted(Path(path).rglob(CFG["source"]["root_marker"]))
    if not hits:
        raise FileNotFoundError(f"no {CFG['source']['root_marker']} under {path}")
    return hits[0].parent


def read_split_lists(root):
    """[(split, 'category/number')] in list order, from the official train/val/test.txt."""
    out = []
    for split in ("train", "val", "test"):
        for line in (root / f"{split}.txt").read_text(encoding="utf-8").split():
            rel = line.strip("/")
            if rel:
                out.append((split, rel))
    return out


def png(arr):
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    assert ok
    return buf.tobytes()


def process(job):
    """One plan -> (metadata row, files {name: bytes}, statistics dict). Never raises."""
    split, rel, root = job
    plan_id = rel.replace("/", "-")
    row = {"plan_id": plan_id, "split": split, "folder": rel, "kept": False, "drop_reason": ""}
    folder = Path(root) / rel
    try:
        plan = parse_svg(folder / "model.svg")
    except Exception as e:  # malformed SVG
        row["drop_reason"] = f"svg: {type(e).__name__}"
        return row, {}, None
    t, n_walls = wall_thickness(plan)
    row.update(n_shapes=len(plan.shapes), n_walls=len(plan.of_kind("wall")), wall_t_svg=t,
               n_bad_elements=plan.n_bad)
    if n_walls < CFG["min_walls_for_thickness"] or not np.isfinite(t) or t <= 0:
        row["drop_reason"] = "no measurable walls"
        return row, {}, None
    try:
        info_s = {}
        scaled = load_grey(folder / "F1_scaled.png", info_s)
    except Exception:
        row["drop_reason"] = "F1_scaled.png unreadable"
        return row, {}, None
    frame = scaled.shape[:2]
    source, src_name = scaled, "F1_scaled"
    try:
        info_o = {}
        orig = load_grey(folder / "F1_original.png", info_o)
        rw, rh = orig.shape[1] / frame[1], orig.shape[0] / frame[0]
        row["orig_ratio"] = rw
        if abs(rw - rh) / rw < 0.02 and rw > 1:     # same drawing, higher resolution
            source, src_name, info_s = orig, "F1_original", info_o
    except Exception:
        row["orig_ratio"] = np.nan

    pcfg, ccfg = CFG["parser_set"], CFG["completion_set"]
    s_par = pcfg["wall_px"] / t
    long_side = max(frame) * s_par
    if long_side > pcfg["max_side"]:
        s_par = pcfg["max_side"] / max(frame)
    s_com = ccfg["wall_px"] / t
    lbl_par, _ = draw_labels(plan, frame, s_par)
    lbl_com, before = draw_labels(plan, frame, s_com)
    img_par = make_image(source, lbl_par.shape)
    img_par, n_white = whiten_padding(img_par, lbl_par)      # baked-in black padding -> white paper
    row.update(had_alpha=info_s.get("had_alpha", False),
               black_padding_share=round(n_white / img_par.size, 4))
    if not np.isin(lbl_com, C16.group("W")).any():
        row["drop_reason"] = "no wall pixels after drawing"
        return row, {}, None

    row.update(kept=True, image_source=src_name, frame_h=frame[0], frame_w=frame[1],
               scale_parser=s_par, wall_px_parser=t * s_par, parser_h=lbl_par.shape[0],
               parser_w=lbl_par.shape[1], scale_completion=s_com, completion_h=lbl_com.shape[0],
               completion_w=lbl_com.shape[1], capped=bool(long_side > pcfg["max_side"]))
    row.update({f"default_space_{k}": v for k, v in plan.default_spaces.items()})

    # ---- untyped plans: "Undefined" covers > 50% of room pixels -> parser set only, rooms = ignore
    rooms = np.isin(lbl_com, C16.group("S"))
    share = untyped_rooms(plan, frame, s_com, lbl_com).sum() / max(int(rooms.sum()), 1)
    untyped = bool(rooms.any() and share > C16.cfg["untyped_max_share"])
    row.update(untyped_share=round(float(share), 4), untyped=untyped, in_completion=not untyped)
    if untyped:
        lbl_par = lbl_par.copy()
        lbl_par[np.isin(lbl_par, C16.group("S"))] = IGNORE
        return row, {f"parser_{split}/{plan_id}_img.png": png(img_par),
                     f"parser_{split}/{plan_id}_lbl.png": png(lbl_par)}, None

    # ---- statistics (completion scale)
    walls = np.array([thickness_length(s.points) for s in plan.of_kind("wall")]).reshape(-1, 2) * s_com
    walls = walls[np.isfinite(walls).all(1) & (walls[:, 1] >= 2 * walls[:, 0])]
    foot = fixture_footprint(plan, frame, s_com)
    stats = {
        "split": split,
        "pix_parser": ps.pixel_counts(lbl_par),
        "pix_completion": ps.pixel_counts(lbl_com),
        "components": ps.component_counts(lbl_com),
        "wall_t": walls[:, 0], "wall_len": walls[:, 1],
        "door_len": np.array([opening_length(s.points) for s in plan.of_kind("door")]) * s_com,
        "window_len": np.array([opening_length(s.points) for s in plan.of_kind("window")]) * s_com,
        "widths": ps.square_widths(lbl_com, C16.group("S") + C16.group("OD") + C16.group("F")),
        "adjacency": ps.space_adjacency(lbl_com, CFG["constants"]["adjacency_max_gap"]),
        "fixture_rooms": ps.fixture_rooms(before, lbl_com),
        "door_fixture": ps.door_fixture_distances(lbl_com),
        "fixture_room": ps.fixture_room_distances(lbl_com),
        "fixture_on_structure": int((foot & np.isin(before, C16.group("BOUNDARY"))).sum()),
        "fixture_pixels": int(foot.sum()),
        "open_hist": ps.pixel_open_hist(lbl_com, C16.group("S") + C16.group("OD") + C16.group("F")),
        "room_areas": ps.room_areas(lbl_com),
        "door_pairs": ps.door_pairs(lbl_com, CFG["completion_set"]["wall_px"] + 2),
        "raw_names": {"ignored": plan.ignored_fixtures,
                      "fixtures": _count([s.name for s in plan.of_kind("fixture")]),
                      "spaces": _count([s.name for s in plan.of_kind("space")])},
    }
    files = {f"parser_{split}/{plan_id}_img.png": png(img_par),
             f"parser_{split}/{plan_id}_lbl.png": png(lbl_par),
             f"completion_{split}/{plan_id}.png": png(lbl_com)}
    return row, files, stats


def _count(names):
    out = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="CubiCasa5K folder (or any parent of train.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N plans of each split (testing)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = find_root(args.root)
    out = Path(args.out)
    (out / "report").mkdir(parents=True, exist_ok=True)
    jobs = read_split_lists(root)
    if args.limit:
        jobs = [j for s in ("train", "val", "test") for j in [j for j in jobs if j[0] == s][: args.limit]]
    print(f"root {root}: {len(jobs)} plans listed", flush=True)

    zips = {}
    rows, all_stats = [], []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (row, files, stats) in enumerate(pool.imap(process, [(s, r, str(root)) for s, r in jobs],
                                                          chunksize=4)):
            rows.append(row)
            if stats is not None:
                stats["plan_id"] = row["plan_id"]
                all_stats.append(stats)
            for name, data in files.items():
                zname, inner = name.split("/", 1)
                if zname not in zips:
                    # PNGs are already compressed: store them without re-compressing
                    zips[zname] = zipfile.ZipFile(out / f"{zname}.zip", "w", zipfile.ZIP_STORED)
                zips[zname].writestr(inner, data)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(jobs)} plans, {time.time() - t0:.0f} s", flush=True)
    for z in zips.values():
        z.close()

    meta = pd.DataFrame(rows)
    meta.to_csv(out / "report" / "metadata.csv", index=False, encoding="utf-8")
    from src.stage2.data.report import save_stats
    save_stats(all_stats, out / "report" / "stats.npz")
    print(f"done in {time.time() - t0:.0f} s: {int(meta.kept.sum())} kept, {int((~meta.kept).sum())} dropped",
          flush=True)
    print(meta[~meta.kept].drop_reason.value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()

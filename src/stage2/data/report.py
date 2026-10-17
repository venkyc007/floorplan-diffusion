"""Turn the build's statistics into the size constants, data_stats.md and data_overlay.png.

    python -m src.stage2.data.report --out <build output dir> [--root <CubiCasa root>]

Writes into <out>/report/:
  constraint_constants.yaml   -> copy to configs/stage2/constraint_constants.yaml
  data_stats.md               -> outputs/stage2/data_stats.md
  data_overlay.png            -> outputs/stage2/data_overlay.png
Constants come from the TRAIN split only (val/test never set a number).
"""

import argparse
import datetime
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.data.plan_stats import SPACE_PAIR_CLASSES
from src.stage2.data.rasterise import make_image
from src.utils import load_config

CFG = load_config("stage2/data")
LIST_KEYS = ("wall_t", "wall_len", "door_len", "window_len", "door_fixture", "fixture_room")


# ------------------------------------------------------------------ pooling

def save_stats(all_stats, path):
    """Pool per-plan statistics into plain arrays (no pickle) in one .npz."""
    arrays = {"plan_ids": np.array([s["plan_id"] for s in all_stats]),
              "split": np.array([s["split"] for s in all_stats])}
    for key in ("pix_parser", "pix_completion", "components"):
        arrays[key] = np.stack([s[key] for s in all_stats]) if all_stats else np.zeros((0, C16.n))
    for key in LIST_KEYS:
        arrays[key] = np.concatenate([np.asarray(s[key], float) for s in all_stats]) if all_stats else np.zeros(0)
        arrays[key + "_split"] = np.concatenate(
            [np.full(len(s[key]), s["split"]) for s in all_stats]) if all_stats else np.zeros(0, str)
    for c in C16.group("S") + C16.group("OD") + C16.group("F"):
        name = C16.names[c]
        per = [(s["split"], w) for s in all_stats for w in s["widths"].get(c, [])]
        arrays[f"width_{name}"] = np.array([w for _, w in per], float)
        arrays[f"width_{name}_split"] = np.array([sp for sp, _ in per])
    for split in ("train", "all"):
        sel = [s for s in all_stats if split == "all" or s["split"] == split]
        k = len(SPACE_PAIR_CLASSES)
        arrays[f"adj_direct_{split}"] = sum((s["adjacency"][0] for s in sel), np.zeros((k, k), np.int64))
        arrays[f"adj_total_{split}"] = sum((s["adjacency"][1] for s in sel), np.zeros((k, k), np.int64))
        arrays[f"fixture_rooms_{split}"] = sum((s["fixture_rooms"] for s in sel),
                                               np.zeros((len(C16.group("F")), C16.n), np.int64))
    arrays["fixture_on_structure"] = np.array(sum(s["fixture_on_structure"] for s in all_stats))
    arrays["fixture_pixels"] = np.array(sum(s["fixture_pixels"] for s in all_stats))
    names = {"ignored": {}, "fixtures": {}, "spaces": {}}
    for s in all_stats:
        for group, counts in s["raw_names"].items():
            for n, v in counts.items():
                names[group][n] = names[group].get(n, 0) + v
    arrays["raw_names_json"] = np.array(json.dumps(names))
    np.savez_compressed(path, **arrays)


# ------------------------------------------------------------------ constants

def wpercentile(values, weights, q):
    """Weighted percentile (q in 0..100)."""
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w) / w.sum()
    return float(v[min(np.searchsorted(cum, q / 100), len(v) - 1)])


def train_only(st, key):
    return st[key][st[key + "_split"] == "train"]


def compute_constants(st):
    c = CFG["constants"]
    vals, prov = {}, {}
    wt, wl = train_only(st, "wall_t"), train_only(st, "wall_len")
    vals["T_wall"] = CFG["completion_set"]["wall_px"]
    prov["T_wall"] = (f"length-weighted median wall-polygon thickness = {wpercentile(wt, wl, 50):.2f} px "
                      f"(3 by construction: every plan is scaled so this is 3)")
    p = wpercentile(wt, wl, c["wall_max_pct"])
    vals["T_wall_max"] = int(np.ceil(p))
    prov["T_wall_max"] = f"ceil(length-weighted p{c['wall_max_pct']} of wall-polygon thickness = {p:.2f} px)"
    for key, src in (("door", "door_len"), ("win", "window_len")):
        x = train_only(st, src)
        lo, hi = np.percentile(x, c["opening_low_pct"]), np.percentile(x, c["opening_high_pct"])
        vals[f"Lmin_{key}"], vals[f"Lmax_{key}"] = int(np.floor(lo)), int(np.ceil(hi))
        prov[f"Lmin_{key}"] = f"floor(p{c['opening_low_pct']} of {src} = {lo:.2f} px), n = {len(x)}"
        prov[f"Lmax_{key}"] = f"ceil(p{c['opening_high_pct']} of {src} = {hi:.2f} px)"
    vals["k"], prov["k"] = {}, {}
    for cls in C16.group("S") + C16.group("OD") + C16.group("F"):
        name = C16.names[cls]
        w = train_only(st, f"width_{name}")
        if len(w) == 0:
            continue
        q = np.percentile(w, c["min_width_pct"])
        vals["k"][name] = max(1, int(np.floor(q)))
        prov["k"][name] = (f"floor(p{c['min_width_pct']} of the largest square fitting in each component "
                           f"= {q:.1f} px), n = {len(w)} components")
    d = train_only(st, "door_fixture")
    d = d[np.isfinite(d)]
    q = np.percentile(d, c["clearance_pct"])
    vals["k_clear"] = max(1, 2 * int(np.floor(q)) - 1)
    prov["k_clear"] = (f"kernel 2*floor(d) - 1, d = p{c['clearance_pct']} of door-to-nearest-fixture "
                       f"distance = {q:.2f} px (so ~{c['clearance_pct']}% of real doors violate), n = {len(d)} doors")
    r = train_only(st, "fixture_room")
    q = np.percentile(r, c["fixture_room_pct"])
    vals["k_fix"] = 2 * int(np.ceil(q)) + 1
    prov["k_fix"] = (f"kernel 2*ceil(r) + 1, r = p{c['fixture_room_pct']} of the farthest fixture pixel's "
                     f"distance to a room = {q:.2f} px, n = {len(r)} fixtures")

    # C6: pairs of spaces that touch directly in real plans
    direct, total = st["adj_direct_train"], st["adj_total_train"]
    names = [C16.names[i] for i in SPACE_PAIR_CLASSES]
    excluded, rates = [], {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if total[i, j] == 0:
                continue
            rate = direct[i, j] / total[i, j]
            rates[f"{names[i]}-{names[j]}"] = round(float(rate), 4)
            if rate > c["contact_exclude_rate"]:
                excluded.append([names[i], names[j]])
    vals["c6_excluded_pairs"] = excluded
    prov["c6_excluded_pairs"] = (f"pairs whose direct 4-neighbour contact is > {c['contact_exclude_rate']:.0%} of "
                                 f"their adjacency (direct + across boundary <= {c['adjacency_max_gap']} px)")
    vals["c6_contact_rates"] = rates

    # C15: compatible rooms per fixture
    fr = st["fixture_rooms_train"].astype(float)
    vals["fixture_rooms"] = {}
    for i, f in enumerate(C16.group("F")):
        share = fr[i] / max(fr[i].sum(), 1)
        vals["fixture_rooms"][C16.names[f]] = [C16.names[k] for k in C16.group("S") + C16.group("OD")
                                               if share[k] >= c["fixture_room_share"]]
    prov["fixture_rooms"] = f"room types holding >= {c['fixture_room_share']:.0%} of the fixture class's pixels"
    return vals, prov


# ------------------------------------------------------------------ report

def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def write_markdown(meta, st, vals, prov, path, n_svgs=None):
    L = [f"# Stage 2 data: CubiCasa5K rasterised from SVG", "",
         f"Built {datetime.date.today()} by `python -m src.stage2.data.build_cubicasa` on Kaggle "
         f"(source: Kaggle dataset `{CFG['source']['kaggle_dataset']}`, the complete mirror of Zenodo 2613548). "
         f"Parser set: greyscale drawing + labels, median wall {CFG['parser_set']['wall_px']} px, long side "
         f"<= {CFG['parser_set']['max_side']}. Completion set: labels only, median wall "
         f"{CFG['completion_set']['wall_px']} px. Official split kept.", "", "## Plans", ""]
    rows = []
    for split in ("train", "val", "test"):
        m = meta[meta.split == split]
        rows.append([split, len(m), int(m.kept.sum()), int((~m.kept).sum())])
    rows.append(["total", len(meta), int(meta.kept.sum()), int((~meta.kept).sum())])
    L.append(table(["split", "listed", "kept", "dropped"], rows))
    if n_svgs is not None:
        L += ["", f"Plan folders with a model.svg: {n_svgs}; listed in the official split files: {len(meta)} "
              f"({n_svgs - len(meta)} not in any list, not used)."]
    drops = meta[~meta.kept]
    L += ["", "**Dropped plans and why:**", ""]
    L.append(table(["reason", "plans", "ids"], [[r, len(g), ", ".join(g.plan_id.head(8))]
                                                 for r, g in drops.groupby("drop_reason")]) if len(drops) else "none")
    k = meta[meta.kept]
    L += ["", "## Scale", "",
          table(["quantity", "median", "p2", "p98", "max"], [
              ["wall thickness in the SVG frame (px)"] + [f"{np.nanpercentile(k.wall_t_svg, q):.1f}" for q in (50, 2, 98, 100)],
              ["parser map long side (px)"] + [f"{np.percentile(np.maximum(k.parser_h, k.parser_w), q):.0f}" for q in (50, 2, 98, 100)],
              ["completion map long side (px)"] + [f"{np.percentile(np.maximum(k.completion_h, k.completion_w), q):.0f}" for q in (50, 2, 98, 100)],
          ]), "",
          f"Plans that hit the 2048-px cap (walls thinner than 8 px in the parser set): {int(k.capped.sum())}"
          + (f"; their wall width there: median {k[k.capped].wall_px_parser.median():.1f} px." if k.capped.any() else "."),
          f"Image source: {(k.image_source == 'F1_original').sum()} plans from F1_original.png (higher resolution), "
          f"{(k.image_source == 'F1_scaled').sum()} from F1_scaled.png."]

    L += ["", "## Class pixel frequencies and components (train split)", ""]
    tr = st["split"] == "train"
    pc, pp, comp = st["pix_completion"][tr], st["pix_parser"][tr], st["components"][tr]
    rows = []
    for c, name in enumerate(C16.names):
        rows.append([name, f"{100 * pc[:, c].sum() / pc.sum():.3f}", f"{100 * pp[:, c].sum() / pp.sum():.3f}",
                     f"{100 * (pc[:, c] > 0).mean():.1f}", int(comp[:, c].sum()), f"{comp[:, c].mean():.2f}"])
    L.append(table(["class", "% pixels (completion)", "% pixels (parser)", "% plans with it",
                    "components (4-conn)", "components per plan"], rows))

    L += ["", "## Size constants (`configs/stage2/constraint_constants.yaml`)", "",
          "Measured on the train split at completion scale (walls 3 px).", ""]
    rows = [[key, vals[key], prov[key]] for key in ("T_wall", "T_wall_max", "Lmin_door", "Lmax_door",
                                                    "Lmin_win", "Lmax_win", "k_clear", "k_fix")]
    rows += [[f"k_{n}", v, prov["k"][n]] for n, v in vals["k"].items()]
    L.append(table(["constant", "value", "how"], rows))
    L += ["", "**C6, spaces that touch directly in real plans** (direct contact / all adjacency; "
          f"pairs above {CFG['constants']['contact_exclude_rate']:.0%} are excluded from C6):", ""]
    rates = sorted(vals["c6_contact_rates"].items(), key=lambda kv: -kv[1])
    L.append(table(["pair", "direct-contact rate", "excluded"],
                   [[p, f"{r:.3f}", "yes" if r > CFG["constants"]["contact_exclude_rate"] else ""] for p, r in rates]))
    L += ["", "**C15, compatible rooms per fixture** (share of the fixture's pixels lying on each room type):", ""]
    fr = st["fixture_rooms_train"].astype(float)
    hdr = [C16.names[c] for c in C16.group("S") + C16.group("OD") + C16.group("BG")]
    rows = []
    for i, f in enumerate(C16.group("F")):
        share = fr[i] / max(fr[i].sum(), 1)
        rows.append([C16.names[f]] + [f"{100 * share[C16.index[h]]:.1f}" for h in hdr]
                    + [", ".join(vals["fixture_rooms"][C16.names[f]])])
    L.append(table(["fixture"] + hdr + ["A(f)"], rows))
    fos, fp = int(st["fixture_on_structure"]), int(st["fixture_pixels"])
    L += ["", f"Fixture outlines overlapping wall/door/window/railing (not painted there): "
          f"{fos} of {fp} fixture pixels ({100 * fos / max(fp, 1):.2f}%)."]

    names = json.loads(str(st["raw_names_json"]))
    L += ["", "## Raw CubiCasa names (all splits)", "",
          "Fixtures painted: " + ", ".join(f"{n} {v}" for n, v in sorted(names["fixtures"].items(), key=lambda kv: -kv[1])),
          "", "Fixtures ignored: " + ", ".join(f"{n} {v}" for n, v in sorted(names["ignored"].items(), key=lambda kv: -kv[1])),
          "", "Space types: " + ", ".join(f"{n} {v}" for n, v in sorted(names["spaces"].items(), key=lambda kv: -kv[1]))]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def overlay_figure(out, meta, path, n_plans, seed):
    """n_plans random kept plans: drawing | parser labels on it | completion labels on it | 1:1 zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    kept = meta[meta.kept]
    pick = kept.sample(n=min(n_plans, len(kept)), random_state=seed)
    cols = 2
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, 4 * cols, figsize=(8 * cols * 2, 4.2 * rows))
    axes = np.atleast_2d(axes)
    zips = {}

    def read(zname, inner, flag):
        if zname not in zips:
            zips[zname] = zipfile.ZipFile(Path(out) / f"{zname}.zip")
        buf = np.frombuffer(zips[zname].read(inner), np.uint8)
        return cv2.imdecode(buf, flag)

    for n, (_, r) in enumerate(pick.iterrows()):
        img = read(f"parser_{r.split}", f"{r.plan_id}_img.png", cv2.IMREAD_GRAYSCALE)
        lbl = read(f"parser_{r.split}", f"{r.plan_id}_lbl.png", cv2.IMREAD_UNCHANGED)
        com = read(f"completion_{r.split}", f"{r.plan_id}.png", cv2.IMREAD_UNCHANGED)
        rgb = np.repeat(img[:, :, None], 3, 2)
        over = (0.45 * rgb + 0.55 * C16.colourise(lbl)).astype(np.uint8)
        small = make_image(img, com.shape)
        over_c = (0.45 * np.repeat(small[:, :, None], 3, 2) + 0.55 * C16.colourise(com)).astype(np.uint8)
        doors = np.argwhere(lbl == C16.index["door"])
        cy, cx = doors[len(doors) // 2] if len(doors) else np.array(lbl.shape) // 2
        h = 90
        y0, x0 = max(0, cy - h), max(0, cx - h)
        zoom = over[y0:y0 + 2 * h, x0:x0 + 2 * h]
        ax = axes[n // cols, (n % cols) * 4:(n % cols) * 4 + 4]
        panels = [(rgb, f"{r.plan_id} ({r.split})"), (over, f"parser {lbl.shape[1]}x{lbl.shape[0]}, walls {r.wall_px_parser:.1f} px"),
                  (over_c, f"completion {com.shape[1]}x{com.shape[0]}, walls 3 px"), (zoom, "parser, 1:1 zoom at a door")]
        for a, (im, title) in zip(ax, panels):
            a.imshow(im, interpolation="nearest")
            a.set_title(title, fontsize=9)
            a.axis("off")
    for a in axes.ravel()[len(pick) * 4:]:
        a.axis("off")
    handles = [Patch(color=C16.palette[c] / 255, label=n) for c, n in enumerate(C16.names)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=11, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=70)
    plt.close(fig)


def write_constants_yaml(vals, prov, n_train, path):
    head = ("# Stage 2 constraint size constants (docs/constraints_spec.md §0), in px at completion scale\n"
            "# (median wall = 3 px). GENERATED by `python -m src.stage2.data.report` from the CubiCasa5K\n"
            f"# train split ({n_train} plans) on {datetime.date.today()}. Do not edit by hand.\n")
    doc = {"placeholder": False, "values": vals, "provenance": prov}
    Path(path).write_text(head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120),
                          encoding="utf-8")


def make_report(out, root=None):
    out = Path(out)
    meta = pd.read_csv(out / "report" / "metadata.csv")
    st = dict(np.load(out / "report" / "stats.npz"))
    vals, prov = compute_constants(st)
    n_svgs = len(list(Path(root).rglob("model.svg"))) if root else None
    write_constants_yaml(vals, prov, int((st["split"] == "train").sum()), out / "report" / "constraint_constants.yaml")
    write_markdown(meta, st, vals, prov, out / "report" / "data_stats.md", n_svgs)
    overlay_figure(out, meta, out / "report" / "data_overlay.png", CFG["report"]["overlay_plans"], CFG["report"]["seed"])
    print((out / "report" / "data_stats.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    make_report(a.out, a.root)

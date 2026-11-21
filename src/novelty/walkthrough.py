"""Novelty - step-by-step pictures of the final hybrid on held-out drawings (for the report / slides).

    python -m src.novelty.walkthrough --zip outputs/kaggle/stage2_data_v3/stage2_data/parser_val.zip --cases 4
        -> outputs/novelty/walkthrough/case_<k>_<drawing>.png  (one picture per case)

What it does: takes the first K held-out drawings in the evaluation's own order (no picking), erases the same LARGE
omission the held-out evaluation erased (same generator, same seeds as src/novelty/eval_e2e.py), runs the final
pipeline (src/novelty/pipeline.run, nothing about the omission given) and draws every stage side by side:
  1 original drawing        2 drawing given (part erased)     3 ink map            4 parser's map
  5 structure net's walls + region flag                        6 after the fill (label map)
  7 the result drawn back on the drawing (run_pipeline's overlay: additions in red, doors as gap + leaf + arc)
  8 ground truth            9 scores
How it fits in: a viewer only. It reads the frozen config, changes no setting and writes nothing but the pictures.
The score of every case is recomputed here with the evaluation's own function (eval_e2e.row_of); on the laptop's CPU
the diffusion finish can differ by a few pixels from the Kaggle GPU run in outputs/novelty/e2e_heldout_v3c_*.csv.
"""

import argparse
import time
import zipfile

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.novelty import pipeline as nov
from src.novelty.eval_e2e import BINS, OUT, drawing_ids, row_of
from src.stage2.data.classes16 import C16
from src.stage2.detect.omission_mix import omit_in_bin
from src.stage2.pipeline import large_fill_compare as lfc

WALL = C16.index["wall"]
LEGEND = ("wall", "door", "window")                     # classes named in the legend (the rest are room types / fixtures)


def large_cases(z, split, k):
    """The first k drawings of the split (evaluation order) that have a large omission -> dicts (same cases as
    eval_e2e: one generator per drawing, small, medium and large drawn in that order)."""
    read = lambda n, f: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), f)
    out = []
    for idx, pid in enumerate(drawing_ids(split, z)):
        lbl = read(f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
        rng = np.random.default_rng([31, idx])
        o = {b: omit_in_bin(b, img, lbl, rng) for b in BINS}["large"]
        if o is not None:
            out.append({"pid": pid, "img": img, "lbl": lbl, "o": o})
        if len(out) == k:
            break
    return out


def crop_of(area, shape, image_shape):
    """Crop around the omission: slices on the map and on the drawing, and the omission box on the map."""
    H, W = shape
    y0, x0, y1, x1 = lfc.box_of(area)
    m = max(24, (y1 - y0) // 2, (x1 - x0) // 2)
    sl = (slice(max(0, y0 - m), min(H, y1 + m + 1)), slice(max(0, x0 - m), min(W, x1 + m + 1)))
    fy, fx = image_shape[0] / H, image_shape[1] / W
    isl = (slice(int(sl[0].start * fy), int(sl[0].stop * fy)), slice(int(sl[1].start * fx), int(sl[1].stop * fx)))
    return sl, isl, (y0, x0, y1, x1), (fy, fx)


def draw(c, res, walls, row, path):
    o, parsed = c["o"], res["parsed"]
    H, W = parsed.shape
    area = lfc.to_parse_scale(o[5], (H, W)) | lfc.to_parse_scale(o[1], (H, W))
    sl, isl, (y0, x0, y1, x1), (fy, fx) = crop_of(area, (H, W), c["img"].shape)
    gt = cv2.resize(c["lbl"], (W, H), interpolation=cv2.INTER_NEAREST)
    ink, _ = nov.drawing_maps(o[0], (H, W))
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.2))
    ax = axes.ravel()

    def box(a, b, colour, style="-", scale=(1, 1), origin=(sl[0].start, sl[1].start), lw=1.6):
        a.add_patch(plt.Rectangle(((b[1] - origin[1]) * scale[1] - .5, (b[0] - origin[0]) * scale[0] - .5),
                                  (b[3] - b[1] + 1) * scale[1], (b[2] - b[0] + 1) * scale[0], fill=False, ec=colour,
                                  lw=lw, ls=style))

    ax[0].imshow(c["img"][isl], cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("1. Original drawing\n(the system never sees it)")
    ax[1].imshow(o[0][isl], cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("2. Drawing GIVEN to the system\n(red dashes = what was erased; for you only)")
    for a in (ax[0], ax[1]):
        box(a, (y0, x0, y1, x1), "red", "--", scale=(fy, fx))
    ax[2].imshow(ink[sl], cmap="gray_r", vmin=0, vmax=255, interpolation="nearest")
    ax[2].set_title("3. Ink map: where the drawing has ink\n(white = nothing drawn there)")
    ax[3].imshow(C16.colourise(parsed[sl]), interpolation="nearest")
    ax[3].set_title("4. Parser's map of the given drawing\n(rooms just merge; nothing looks wrong)")
    img5 = C16.colourise(parsed[sl]).copy()
    if walls is not None:
        img5[walls[sl]] = (255, 0, 255)
    ax[4].imshow(img5, interpolation="nearest")
    regions = [h for h in res["holes"] if h["type"] == "region"]
    for h in regions:
        box(ax[4], h["box"], "blue", lw=2.2)
    ax[4].set_title(f"5. Structure net: missing walls (magenta)\n{len(regions)} region flag(s) on the plan (blue box)")
    ax[5].imshow(C16.colourise(res["completed"][sl]), interpolation="nearest")
    ax[5].set_title("6. After the fill\n(the completed map)")
    ax[6].imshow(res["overlay"][isl])
    ax[6].set_title("7. Result drawn back on the drawing\n(red = what the system added)")
    ax[7].imshow(C16.colourise(gt[sl]), interpolation="nearest")
    ax[7].set_title("8. Ground truth\n(what was really there)")
    for a in ax[:8]:
        a.title.set_fontsize(10)
    for a in (ax[3], ax[4], ax[5], ax[7]):
        box(a, (y0, x0, y1, x1), "red", "--", lw=1.0)
    for a in ax:
        a.axis("off")
    yes = lambda v: "YES" if v else "NO"
    num = lambda v: "-" if v == "" else f"{float(v):.2f}"
    lines = [f"drawing {c['pid']}", f"erased: {row['kind']}, {row['size_pct']}% of the building", "",
             f"noticed by the system:      {yes(row['found'])}",
             f"flagged as a missing region: {yes(row['region_flag'])}", "",
             f"recovered (2 px rule):       {yes(row['recovered'])}",
             f"   missing walls brought back: {num(row['miss_recall'])}",
             f"   added walls that are right: {num(row['add_precision'])}", "",
             f"fixed (strict pixel rule):   {yes(row['fixed'])}",
             f"   pixel overlap (IoU): {num(row['fix_iou'])}  (needs 0.30)"]
    ax[8].text(0.0, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10.5)
    for i, name in enumerate(LEGEND):
        ax[9].add_patch(plt.Rectangle((0.02, 0.74 - 0.09 * i), 0.07, 0.06, fc=C16.palette[C16.index[name]] / 255, ec="k",
                                      transform=ax[9].transAxes))
        ax[9].text(0.12, 0.77 - 0.09 * i, name, va="center", transform=ax[9].transAxes, fontsize=11)
    ax[9].text(0.02, 0.98, "colours in the maps\n(other colours = room types, fixtures)", va="top",
               transform=ax[9].transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="parser_val.zip (drawings + labels of the val split)")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--split", default="heldout", choices=("heldout", "tuning"))
    ap.add_argument("--tag", default="", help="suffix of the output folder (e.g. _valid for another NOVELTY_CONFIG)")
    args = ap.parse_args()
    out = OUT / f"walkthrough{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(large_cases(zipfile.ZipFile(args.zip), args.split, args.cases)):
        o = c["o"]
        t = time.time()
        res = nov.run(o[0])
        walls = nov.STATE.get("walls")
        oc = {0: o[0], 1: o[1], 2: o[2], 4: o[4], 5: o[5], "lbl": c["lbl"]}
        row = row_of(c["pid"], "large", oc, "hybrid auto", res, time.time() - t)
        path = out / f"case_{k + 1}_{c['pid']}.png"
        draw(c, res, walls, row, path)
        print({f: row[f] for f in ("pid", "kind", "size_pct", "found", "region_flag", "recovered", "fixed", "fix_iou",
                                    "seconds")}, "->", path, flush=True)


if __name__ == "__main__":
    main()

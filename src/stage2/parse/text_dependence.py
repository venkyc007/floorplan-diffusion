"""Does the parser read the room NAMES to decide room types? Parser v1 on drawings with and without text.

    python -m src.stage2.parse.text_dependence --zip parser_val.zip [--n 100]
        -> outputs/stage2/text_dependence.md, outputs/stage2/text_erasure_examples.png

Drawings: the first n typed val drawings in file order. Text erasure: dark connected pieces (ink < 170,
8-connected) of character size (bounding box <= 4 wall thicknesses = 32 px each side, <= 400 px, at least
3 px on one side) lying inside rooms (>= 95% on room-type labels) and not touching walls, openings, railings
or fixtures in the label map (grown 2 px), inpainted away with src/stage2/data/erasure.py. Control: wall /
door / window IoU, original vs erased (should not move if only text was removed). Room labels ("MH", "KPH 3,3 m2", ...) are such pieces;
furniture outlines and hatching are mostly larger or touch walls and stay.
Parser v1 at scale 1 (the parser set already has ~8 px walls), same as its validation. Scores pooled over
all drawings: IoU of the 6 room types and their mean, and the room-type confusion matrix (rows = true
type, columns = predicted type or "not a room"; % of the row's pixels), original vs text-erased.
"""

import argparse
import zipfile

import cv2
import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import erase, grown
from src.stage2.pipeline.run import parse_at
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
ROOMS = C16.group("S")
WALL_PX = 8


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def text_mask(img, lbl, ink=170, max_side=4 * WALL_PX, max_area=400, inside=0.95):
    """Character-sized dark pieces lying inside rooms (>= 95% of their pixels on room-type labels; room names
    sit inside rooms), not touching walls / openings / railings / fixtures. ink 170 also takes the light
    anti-aliased letter edges (a first version at 128 left letter ghosts)."""
    n, comp, stats, _ = cv2.connectedComponentsWithStats((img < ink).astype(np.uint8), connectivity=8)
    w, h, a = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    ok = (w <= max_side) & (h <= max_side) & (a <= max_area) & (np.maximum(w, h) >= 3)
    ok[0] = False
    solid = grown(np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")), 2)
    ok[np.unique(comp[solid])] = False
    in_room = np.bincount(comp[np.isin(lbl, C16.group("S"))].ravel(), minlength=n)
    ok &= in_room >= inside * a
    return ok[comp]


def confusion(pred, gt):
    """(6, 7) pixel counts: true room type x (predicted room type, or not a room)."""
    m = np.zeros((len(ROOMS), len(ROOMS) + 1), np.int64)
    for i, c in enumerate(ROOMS):
        p = pred[gt == c]
        for j, d in enumerate(ROOMS):
            m[i, j] = (p == d).sum()
        m[i, -1] = len(p) - m[i, :-1].sum()
    return m


def ious(conf, pred_counts):
    """Per room type IoU from the confusion counts + predicted pixel counts per room type."""
    out = []
    for i in range(len(ROOMS)):
        inter = conf[i, i]
        union = conf[i].sum() + pred_counts[i] - inter
        out.append(inter / union if union else np.nan)
    return np.array(out)


def table(conf):
    names = [C16.names[c] for c in ROOMS]
    L = ["| true \\ predicted | " + " | ".join(names) + " | not a room |", "|---|" + "---|" * (len(names) + 1)]
    for i, n in enumerate(names):
        row = conf[i] / max(conf[i].sum(), 1)
        L.append(f"| {n} | " + " | ".join(f"{100 * v:.0f}" for v in row) + " |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    conf = {"original": 0, "text erased": 0}
    STRUCT = [C16.index[c] for c in ("wall", "door", "window")]
    sio = {k: np.zeros((len(STRUCT), 2), np.int64) for k in conf}       # (inter, union) per class
    pcount = {"original": np.zeros(len(ROOMS), np.int64), "text erased": np.zeros(len(ROOMS), np.int64)}
    examples, done, erased_px = [], 0, []
    for n in names:
        if done >= args.n:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        tm = text_mask(img, lbl)
        clean = erase(img, tm, lbl, wall_px=WALL_PX)
        erased_px.append(int(tm.sum()))
        for name, im in (("original", img), ("text erased", clean)):
            pred, _ = parse_at(im, 1.0, 2048)
            gt = cv2.resize(lbl, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) \
                if pred.shape != lbl.shape else lbl
            conf[name] = conf[name] + confusion(pred, gt)
            pcount[name] += np.array([(pred == c).sum() for c in ROOMS])
            for i, c in enumerate(STRUCT):
                sio[name][i] += [((pred == c) & (gt == c)).sum(), ((pred == c) | (gt == c)).sum()]
        if len(examples) < 4 and tm.sum() > 200:
            ys, xs = np.nonzero(tm)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            y0, x0 = max(0, cy - 200), max(0, cx - 200)
            examples.append((n, img[y0:y0 + 400, x0:x0 + 400], clean[y0:y0 + 400, x0:x0 + 400]))
        done += 1
        print(f"{done}/{args.n} {n}: {tm.sum()} text px erased", flush=True)
    res = {k: ious(conf[k], pcount[k]) for k in conf}
    drop = 100 * (np.nanmean(res["original"]) - np.nanmean(res["text erased"]))
    names_r = [C16.names[c] for c in ROOMS]
    L = ["# Room types: does the parser read the text? (parser v1, val)", "",
         f"{done} typed val drawings (file order). Text erased: dark character-sized pieces not touching walls, "
         f"openings, railings or fixtures, inpainted (median {int(np.median(erased_px))} px per drawing). "
         "Examples: outputs/stage2/text_erasure_examples.png.", "",
         "| room type IoU | " + " | ".join(names_r) + " | mean |", "|---|" + "---|" * (len(names_r) + 1)]
    for k in res:
        L.append(f"| {k} | " + " | ".join(f"{100 * v:.1f}" for v in res[k]) + f" | {100 * np.nanmean(res[k]):.1f} |")
    L += ["", "Control (did the erasure damage structure?): " + "; ".join(
        f"{k}: wall / door / window IoU " + " / ".join(f"{100 * a / max(b, 1):.1f}" for a, b in sio[k]) for k in sio) + "."]
    L += ["", f"**Room-type mIoU drop when text is erased: {drop:.1f} points.**", "",
          "## Confusion, original (% of each true type's pixels)", ""] + table(conf["original"]) + \
         ["", "## Confusion, text erased", ""] + table(conf["text erased"])
    if drop > 5:
        L += ["", "## Proposal (not launched): parser v2", "",
              "The parser leans on room names. Proposal: parser v2 = fine-tune v1 with random text erasure as an "
              "augmentation (this module's text mask + inpainting, applied to ~50% of training drawings) plus the "
              "local constraint losses, as planned for v2 in CLAUDE.md. Same data and split; select on val room-type "
              "mIoU measured on text-erased drawings."]
    (OUT / "text_dependence.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 5 * len(examples)), squeeze=False)
    for (n, a, b), (l, r) in zip(examples, axes):
        l.imshow(a, cmap="gray", vmin=0, vmax=255)
        r.imshow(b, cmap="gray", vmin=0, vmax=255)
        l.set_title(f"{n}: original", fontsize=9)
        r.set_title("text erased", fontsize=9)
        l.axis("off")
        r.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "text_erasure_examples.png", dpi=80)


if __name__ == "__main__":
    main()

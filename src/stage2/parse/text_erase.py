"""Text eraser for drawings: removes room names / areas as whole WORDS and LINES (parser v2, 2026-10-19).

    mask = text_mask(img, lbl)            # lbl: ground-truth labels (training, evaluation) or a parse (inference)
    clean = erase_text(img, lbl)          # text inpainted away (src/stage2/data/erasure.py)
    python -m src.stage2.parse.text_erase --zip parser_val.zip   -> outputs/stage2/text_erasure_examples.png

1. Character candidates: dark connected pieces (ink < 170, 8-connected) of character size (bounding box
   <= 4 wall thicknesses = 32 px each side, <= 400 px, at least 3 px on one side), not touching walls,
   openings, railings or fixtures (label map grown 2 px).
2. Words / lines: candidates are joined when they lie within ~1 character height of each other (dilation by
   GAP px), so a label is taken whole or not at all (the per-character version left half-erased words where
   one letter crossed a room boundary or two letters touched).
3. A group is text if >= 80% of its candidate pixels lie on room-type labels (names sit inside rooms) and it
   is no bigger than MAX_GROUP px (long dimension lines / hatching made of short dashes stay).
4. Inside each accepted group's box (grown 2 px), every dark pixel not on a protected component is erased,
   so joined letters, underscores and dots of the label go too.
"""

import argparse
import zipfile

import cv2
import numpy as np

from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import erase, grown
from src.utils import PROJECT_ROOT

WALL_PX = 8
INK = 170
GAP = 10                 # px: letters / words closer than this join one group (large fonts space wide)
MAX_GROUP = 6000         # px of candidate ink in one group (a few lines of text)
IN_ROOM = 0.8


def text_mask(img, lbl, ink=INK, max_side=4 * WALL_PX, max_area=400, gap=GAP, inside=IN_ROOM):
    dark = (img < ink).astype(np.uint8)
    n, comp, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    w, h, a = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    ok = (w <= max_side) & (h <= max_side) & (a <= max_area) & (np.maximum(w, h) >= 3)
    ok[0] = False
    solid = grown(np.isin(lbl, C16.group("BOUNDARY") + C16.group("F")), 2)
    ok[np.unique(comp[solid])] = False
    cand = ok[comp]
    if not cand.any():
        return cand
    joined = cv2.dilate(cand.astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
    g, groups, gstats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    room = np.isin(lbl, C16.group("S"))
    gid = groups[cand]
    tot = np.bincount(gid, minlength=g)
    inr = np.bincount(gid, weights=room[cand], minlength=g)
    out = np.zeros_like(cand)
    for i in np.nonzero((tot > 0) & (tot <= MAX_GROUP) & (inr >= inside * np.maximum(tot, 1)))[0]:
        x, y, bw, bh = gstats[i, :4]
        y0, y1, x0, x1 = max(0, y + gap - 2), y + bh - gap + 2, max(0, x + gap - 2), x + bw - gap + 2
        out[y0:y1, x0:x1] |= (dark[y0:y1, x0:x1] > 0) & ~solid[y0:y1, x0:x1]
    return out


def erase_text(img, lbl):
    return erase(img, text_mask(img, lbl), lbl, wall_px=WALL_PX)


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def main():
    """4 before/after crops around the densest text of the first typed val drawings."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    ex = []
    for n in names:
        if len(ex) >= 4:
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        m = text_mask(img, lbl)
        if m.sum() < 300:
            continue
        clean = erase(img, m, lbl, wall_px=WALL_PX)
        dens = cv2.boxFilter(m.astype(np.float32), -1, (301, 301))
        cy, cx = np.unravel_index(int(dens.argmax()), dens.shape)
        y0, x0 = max(0, cy - 200), max(0, cx - 250)
        ex.append((n, img[y0:y0 + 400, x0:x0 + 500], clean[y0:y0 + 400, x0:x0 + 500]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(ex), 2, figsize=(12, 4.9 * len(ex)), squeeze=False)
    for (n, a, b), (l, r) in zip(ex, axes):
        l.imshow(a, cmap="gray", vmin=0, vmax=255)
        r.imshow(b, cmap="gray", vmin=0, vmax=255)
        l.set_title(f"{n}: original", fontsize=9)
        r.set_title("text erased (whole words / lines)", fontsize=9)
        l.axis("off")
        r.axis("off")
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / "stage2" / "text_erasure_examples.png"
    fig.savefig(out, dpi=80)
    print("saved", out)


if __name__ == "__main__":
    main()

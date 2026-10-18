"""Visual check of the realistic erasure: 6 zoomed crops, original vs erased.

    python -m src.stage2.data.erasure_check --zip parser_val.zip   -> outputs/stage2/erasure_check.png

Rows: 2 drawings with grey-shaded rooms, 2 with white rooms (a fixture, else a window, erased), 2 door
kind (a) (wall drawn across the doorway). Drawings in file order (typed val plans); a drawing counts as
grey-shaded if the median grey of its room pixels is below 235.
"""

import argparse
import zipfile

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import door_symbol, draw_wall, erase
from src.utils import PROJECT_ROOT

I = C16.index


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def first(lbl, classes, min_px=40):
    comps = [m for m in components(np.isin(lbl, classes)) if m.sum() >= min_px]
    return comps[0] if comps else None


def crop(img, m, pad=60):
    ys, xs = np.nonzero(m)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    want = {"grey": 2, "white": 2, "door (a)": 2}
    rows = []
    for n in names:
        if not any(want.values()):
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        rooms = np.isin(lbl, C16.group("S"))
        shade = "grey" if rooms.any() and np.median(img[rooms]) < 235 else "white"
        if want[shade]:
            m = first(lbl, C16.group("F")) if first(lbl, C16.group("F")) is not None else first(lbl, [I["window"]])
            if m is not None:
                rows.append((f"{n} ({shade} rooms): {C16.names[int(np.bincount(lbl[m]).argmax())]} erased",
                             crop(img, m), crop(erase(img, m, lbl), m)))
                want[shade] -= 1
                continue
        if want["door (a)"]:
            d = first(lbl, [I["door"]])
            if d is not None:
                rows.append((f"{n}: door kind (a), wall drawn across", crop(img, d), crop(draw_wall(img, d, lbl), d)))
                want["door (a)"] -= 1
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.2 * len(rows)))
    for (title, a, b), (l, r) in zip(rows, axes):
        l.imshow(a, cmap="gray", vmin=0, vmax=255)
        r.imshow(b, cmap="gray", vmin=0, vmax=255)
        l.set_title("original: " + title, fontsize=8)
        r.set_title("erased (this exact image goes to the pipeline)", fontsize=8)
        l.axis("off")
        r.axis("off")
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / "stage2" / "erasure_check.png"
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()

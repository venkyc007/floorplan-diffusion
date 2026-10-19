"""Visual check of the realistic erasure: 8 zoomed crops, original vs erased.

    python -m src.stage2.data.erasure_check --zip parser_val.zip   -> outputs/stage2/erasure_check.png

Rows: 2 door kind (a) (zone cleared, wall drawn across), 2 door kind (b) (zone cleared, gap left),
2 fixtures and 2 windows (symbol box grown 40% cleared; window: wall drawn across). Typed val drawings in
file order; one example per drawing; fixtures alternate a bath/shower first (the hardest symbol), then any.
The dashed box on the ORIGINAL marks the cleared zone (figure only, never in the pipeline's image).
"""

import argparse
import zipfile

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.data.erasure import door_zone, draw_wall, erase_symbol, open_gap, symbol_zone

I = C16.index
from src.utils import PROJECT_ROOT  # noqa: E402


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def first(lbl, classes, min_px=40):
    comps = [m for m in components(np.isin(lbl, classes)) if m.sum() >= min_px]
    return comps[0] if comps else None


def window(zone, shape, pad=40):
    ys, xs = np.nonzero(zone)
    return max(0, ys.min() - pad), min(shape[0], ys.max() + pad), max(0, xs.min() - pad), min(shape[1], xs.max() + pad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    want = {"door (a)": 2, "door (b)": 2, "fixture": 2, "window": 2}
    rows = []
    for n in names:
        if not any(want.values()):
            break
        lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
        if (lbl == 255).any():
            continue
        img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
        for kind in want:
            if not want[kind]:
                continue
            if kind.startswith("door"):
                m = first(lbl, [I["door"]])
                zone = None if m is None else door_zone(m, img.shape)
                f = draw_wall if kind == "door (a)" else open_gap
            elif kind == "fixture":
                m = first(lbl, [I["bath_shower"]]) if want[kind] == 2 else first(lbl, C16.group("F"))
                zone, f = (None if m is None else symbol_zone(m, img.shape)), erase_symbol
            else:
                m = first(lbl, [I["window"]])
                zone, f = (None if m is None else symbol_zone(m, img.shape)), erase_symbol
            if m is None:
                continue
            name = C16.names[int(np.bincount(lbl[m]).argmax())]
            rows.append((f"{n}: {kind} ({name})", img, f(img, m, lbl), window(zone, img.shape), zone))
            want[kind] -= 1
            break
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.6 * len(rows)))
    for (title, a, b, (y0, y1, x0, x1), zone), (l, r) in zip(rows, axes):
        l.imshow(a[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        r.imshow(b[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        ys, xs = np.nonzero(zone)
        l.add_patch(Rectangle((xs.min() - x0 - 0.5, ys.min() - y0 - 0.5), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1,
                              fill=False, ec="orange", lw=1.2, ls="--"))
        l.set_title("original: " + title + " (dashed: cleared zone)", fontsize=8)
        r.set_title("erased (this exact image goes to the pipeline)", fontsize=8)
        l.axis("off")
        r.axis("off")
    fig.tight_layout()
    out = PROJECT_ROOT / "outputs" / "stage2" / "erasure_check.png"
    fig.savefig(out, dpi=90)
    print("saved", out)


if __name__ == "__main__":
    main()

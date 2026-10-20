"""Figure: 4 examples of the large omissions (2 room (whole), 2 region), original | erased (user, 2026-10-20).

    python -m src.stage2.pipeline.erasure_large_examples --zip parser_val.zip
        -> outputs/stage2/erasure_large_examples.png

Drawings: TUNING val drawings in file order (never held-out ones, so batch 3 stays fresh), the first typed
drawings each kind works on; seed 7. Orange outline = the area mask (on the axes only); title = the metadata.
Light: erasure only, no model (fine on the laptop).
"""

import argparse
import zipfile

import cv2
import numpy as np

from src.stage2.detect.image_eval import LARGE_KINDS, omit
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
TUNING = PROJECT_ROOT / "configs" / "stage2" / "val_split" / "tuning_ids.txt"


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = zipfile.ZipFile(args.zip)
    ids = TUNING.read_text(encoding="utf-8").split()
    rng = np.random.default_rng(7)
    rows, used = [], set()
    for kind in LARGE_KINDS:
        got = 0
        for pid in ids:
            if got == 2:
                break
            if pid in used:
                continue
            lbl = read(z, f"{pid}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():
                continue
            img = read(z, f"{pid}_img.png", cv2.IMREAD_GRAYSCALE)
            o = omit(kind, img, lbl, rng)
            if o is None:
                continue
            used.add(pid)
            got += 1
            rows.append((pid, kind, img, o))
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 7 * len(rows)), squeeze=False)
    for (pid, kind, img, (out, structure, _, _, meta)), (a, b) in zip(rows, axes):
        for ax, im, t in ((a, img, "original"), (b, out, "erased")):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.contour(structure, levels=[0.5], colors="orange", linewidths=0.6)
            ax.set_title(f"{pid}: {kind}, {t}\n{meta['location']}, {meta['position']}, {meta['size_bin']} "
                         f"({meta['size_pct']}% of the building); orange = erased components", fontsize=9)
            ax.axis("off")
        print(pid, kind, meta)
    fig.tight_layout()
    fig.savefig(OUT / "erasure_large_examples.png", dpi=60)


if __name__ == "__main__":
    main()

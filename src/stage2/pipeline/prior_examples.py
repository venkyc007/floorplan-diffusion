"""Sketch-prior examples: the fill of one hole with and without the type-guided sketch (user, 2026-10-19).

    python -m src.stage2.pipeline.prior_examples --zip parser_val.zip   -> outputs/stage2/sketch_prior_examples.png

Cases: the check set's first door (a), door (b), window and fixture omissions (check_set.make_case, erasure v2).
The erased drawing is parsed; the hole is the omission's box (resized to the parse, grown 3 px) with its true type,
so the figure shows the FILL only (detection is evaluated separately). Fast mode (N = 2, 15 steps from t0 = 300),
same seed for both runs; then the type restriction. Columns: parse + hole, fill without prior, fill with prior.
"""

import argparse
import zipfile

import cv2
import numpy as np
import torch
import yaml

from src.stage2.data.classes16 import C16
from src.stage2.detect.gaps import closed_pairs
from src.stage2.pipeline import check_set
from src.stage2.pipeline.run import cfg, fill_hole, parse, restrict
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
KINDS = [("door (a)", "door", ("C8",)), ("door (b)", "door", ("R5",)), ("window", "window", ()), ("fixture", None, ())]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    g = yaml.safe_load((PROJECT_ROOT / "configs" / "stage2" / "detector_threshold.yaml").read_text(encoding="utf-8"))
    g = g.get("gap_rule", {})
    bad = closed_pairs(open_rate=g.get("open_rate", 0.10))
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    c = cfg()
    rows, used = [], set()
    for kind, typ, rules in KINDS:
        for n in names:
            if n in used:
                continue
            lbl = check_set.read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():
                continue
            img = check_set.read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
            case = check_set.make_case(kind, img, lbl, bad, g.get("min_len", 0) * 8 / 3)
            if case is None:
                continue
            used.add(n)
            erased, removed, expected = case
            label, _, _ = parse(erased, c["parse"])
            rm = cv2.resize(removed.astype(np.uint8), (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            ys, xs = np.nonzero(rm)
            box = (max(0, ys.min() - 3), max(0, xs.min() - 3), ys.max() + 3, xs.max() + 3)
            t = typ or expected
            res = {}
            for on in (False, True):
                fc = dict(c["fill"], sketch_prior=on)
                new, _ = fill_hole(label, box, fc, torch.Generator().manual_seed(0), c["fast"], t, rules)
                res[on] = restrict(label, new, t)
            rows.append((f"{n}: {kind} (hole type {t})", label, res[False], res[True], box, rm))
            print(n, kind, "added without / with prior:", int((res[False] != label).sum()), int((res[True] != label).sum()),
                  flush=True)
            break
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4.2 * len(rows)), squeeze=False)
    for (title, label, a, b, box, rm), ax in zip(rows, axes):
        y0, x0, y1, x1 = box
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        sl = (slice(max(0, cy - 40), cy + 40), slice(max(0, cx - 40), cx + 40))
        for axx, m, t in zip(ax, (label, a, b), ("parse + hole (red), omission (orange)", "fill WITHOUT sketch prior",
                                                  "fill WITH sketch prior")):
            axx.imshow(C16.colourise(m[sl]), interpolation="nearest")
            axx.add_patch(Rectangle((x0 - sl[1].start - 0.5, y0 - sl[0].start - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                                    fill=False, ec="red", lw=1))
            if axx is ax[0]:                              # outline only on the parse (it hides the fill)
                axx.contour(rm[sl], levels=[0.5], colors="orange", linewidths=0.8)
            axx.set_title((title + "\n" if axx is ax[0] else "") + t, fontsize=8)
            axx.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "sketch_prior_examples.png", dpi=80)
    print("saved")


if __name__ == "__main__":
    main()

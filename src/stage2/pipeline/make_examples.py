"""Pipeline examples: 3 val drawings with one component whitened out of the IMAGE, run end to end.

    python -m src.stage2.pipeline.make_examples --zip parser_val.zip  -> outputs/stage2/pipeline_examples.png (+ .md)

Plans are taken in file order (no cherry-picking): the first val drawings whose labels contain the
component kind needed (a door, then a window, then a fixture), one example each. The component's box
is whitened in the drawing; for a door the box is grown by the door length on both sides so the
swing arc disappears too. Each drawing runs through run_pipeline() in automatic mode on the CPU.
"""

import argparse
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from src.stage2.constraints.rules import components
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import run_pipeline
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
I = C16.index
KINDS = [("door", [I["door"]]), ("window", [I["window"]]), ("fixture", C16.group("F"))]


def read(z, name, flag):
    return cv2.imdecode(np.frombuffer(z.read(name), np.uint8), flag)


def whiten(img, lbl, classes, kind):
    comps = [m for m in components(np.isin(lbl, classes)) if m.sum() >= 40]
    if not comps:
        return None
    m = comps[0]
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    g = max(y1 - y0, x1 - x0) if kind == "door" else 3
    out = img.copy()
    out[max(0, y0 - g):y1 + g + 1, max(0, x0 - g):x1 + g + 1] = 255
    return out, (int(y0), int(x0), int(y1), int(x1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", required=True)
    args = ap.parse_args()
    z = zipfile.ZipFile(args.zip)
    names = sorted(n[:-8] for n in z.namelist() if n.endswith("_img.png"))
    examples, i = [], 0
    for kind, classes in KINDS:
        while i < len(names):
            n = names[i]
            i += 1
            img = read(z, f"{n}_img.png", cv2.IMREAD_GRAYSCALE)
            lbl = read(z, f"{n}_lbl.png", cv2.IMREAD_UNCHANGED)
            if (lbl == 255).any():                  # untyped plan: skip
                continue
            w = whiten(img, lbl, classes, kind)
            if w is None:
                continue
            t = time.time()
            res = run_pipeline(w[0])
            res["seconds"] = time.time() - t
            examples.append((n, kind, img, w[0], w[1], res))
            print(n, kind, res["timings"], f"{res['seconds']:.0f} s", res["messages"], flush=True)
            break
    figure(examples)


def figure(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(len(examples), 3, figsize=(18, 6.5 * len(examples)))
    lines = ["# Pipeline examples (CPU)", ""]
    for row, (n, kind, img, white, box, res) in zip(np.atleast_2d(axes), examples):
        a, b, c = row
        a.imshow(white, cmap="gray", vmin=0, vmax=255)
        y0, x0, y1, x1 = box
        a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="orange", lw=1.5))
        a.set_title(f"{n[-30:]}: input, {kind} whitened out (orange)", fontsize=9)
        b.imshow(C16.colourise(res["parsed"]), interpolation="nearest")
        for h in res["holes"]:
            hy0, hx0, hy1, hx1 = h["box"]
            b.add_patch(Rectangle((hx0, hy0), hx1 - hx0, hy1 - hy0, fill=False, ec="red", lw=1.2))
            b.text(hx0, hy0 - 2, h["type"], color="red", fontsize=7)
        b.set_title(f"parsed (walls 3 px) + detected holes ({len(res['holes'])})", fontsize=9)
        c.imshow(res["overlay"])
        c.set_title("result: additions in red on the drawing", fontsize=9)
        c.text(0, res["overlay"].shape[0] * 1.02, "\n".join(res["messages"][:6]), fontsize=8, va="top")
        lines += [f"## {n} ({kind} whitened)", "", f"CPU time {res['seconds']:.0f} s ({res['timings']}); "
                  f"parse scale {res['parse_info']['parse_scale']:.2f}, walls {res['parse_info']['wall_px_at_parse_scale']:.1f} px.", ""]
        lines += [f"- {m}" for m in res["messages"]] + [""]
    for ax in np.atleast_2d(axes).ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_examples.png", dpi=80)
    (OUT / "pipeline_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print("saved", OUT / "pipeline_examples.png")


if __name__ == "__main__":
    main()

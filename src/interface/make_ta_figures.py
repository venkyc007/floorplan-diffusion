"""Figures of the final method for the TA meeting (val gallery only, seed 0, frozen app settings).

    python -m src.interface.make_ta_figures            # both figures
    python -m src.interface.make_ta_figures --only mixed|large

  mixed -> outputs/interface/hybrid_samples.png: the first 2 small, 2 medium and 2 large
           gallery cases, one row each (portrait).
  large -> outputs/interface/hybrid_samples_large.png: the FIRST 4 large-hole gallery cases
           (file order, no selection), a 2 x 2 grid of plans (landscape).
Every plan is shown as: input with grey hole | matched U-Net sketch | size-adaptive hybrid |
ground truth, with a dashed box around the hole. Takes ~1 min per plan on the laptop CPU.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data.corruptions import UNKNOWN  # noqa: E402
from src.eval.figures import colorize  # noqa: E402
from src.interface.inference import Repairer, load_config  # noqa: E402
from src.utils import project_path  # noqa: E402

COLS = ["Input (grey = hole)", "Matched U-Net sketch", "Size-adaptive hybrid", "Ground truth"]
HOLE_EDGE = "#d6336c"


def run_case(rep, g, i):
    """The 4 label maps of one gallery case, and the hybrid's details (size bin, t0)."""
    plan, mask, gt = g["inputs"][i], g["masks"][i], g["targets"][i]
    sketch = rep.repair(plan, mask, "unet_matched", seed=0)
    hybrid = rep.repair(plan, mask, "size_adaptive_hybrid", seed=0)
    shown = plan.copy()
    shown[mask] = UNKNOWN                                    # drawn grey
    d = hybrid.details
    print(f"plan {g['plan_ids'][i]}: {g['types'][i]}, {d['size_bin']} ({d['mask_pct_of_building']:.1f}%), "
          f"t0 {d['t0']}, {hybrid.seconds:.1f} s", flush=True)
    return [shown, sketch.labels, hybrid.labels, gt], d


def draw(ax, img, mask, title=None):
    ax.imshow(colorize(img.astype(np.uint8)), interpolation="nearest")
    ys, xs = np.nonzero(mask)
    ax.add_patch(plt.Rectangle((xs.min() - 0.5, ys.min() - 0.5), xs.max() - xs.min() + 1,
                               ys.max() - ys.min() + 1, fill=False, ec=HOLE_EDGE, lw=0.9, ls="--"))
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9.5)


def mixed_figure(rep, g, out):
    cases = [int(i) for b in ("small", "medium", "large") for i in np.flatnonzero(g["size_bins"] == b)[:2]]
    fig, axes = plt.subplots(len(cases), 4, figsize=(4 * 2.6, len(cases) * 2.75))
    for r, i in enumerate(cases):
        images, d = run_case(rep, g, i)
        for c, img in enumerate(images):
            draw(axes[r, c], img, g["masks"][i], COLS[c] if r == 0 else None)
        axes[r, 0].set_ylabel(f"{g['types'][i]} removed\n{d['size_bin']} hole "
                              f"({d['mask_pct_of_building']:.1f}%)\nt₀ = {d['t0']}", fontsize=8.5)
    fig.suptitle("Final method: size-adaptive hybrid (val gallery, seed 0; t₀ = 400 / 600 / 700 for small / "
                 "medium / large holes)\nDashed box = repaired region", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def large_figure(rep, g, out):
    cases = [int(i) for i in np.flatnonzero(g["size_bins"] == "large")[:4]]     # first 4, file order
    fig = plt.figure(figsize=(20, 6.4))
    gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 1, 1, 0.18, 1, 1, 1, 1], left=0.01, right=0.99,
                          top=0.86, bottom=0.01, wspace=0.04, hspace=0.32)   # column 4 = spacer
    for k, i in enumerate(cases):
        row, side = divmod(k, 2)
        col0 = 0 if side == 0 else 5
        images, d = run_case(rep, g, i)
        axes = [fig.add_subplot(gs[row, col0 + c]) for c in range(4)]
        for ax, img, title in zip(axes, images, COLS):
            draw(ax, img, g["masks"][i], title)
        left, right, top = axes[0].get_position().x0, axes[3].get_position().x1, axes[0].get_position().y1
        fig.text((left + right) / 2, top + 0.045, f"Plan {g['plan_ids'][i]} · {g['types'][i]} removed · large "
                 f"hole ({d['mask_pct_of_building']:.1f}% of the building) · t₀ = {d['t0']}",
                 ha="center", fontsize=11.5, weight="bold")
    fig.suptitle("Final method on large holes: size-adaptive hybrid (first 4 large-hole val-gallery plans, "
                 "seed 0, no selection) — dashed box = repaired region", fontsize=13, y=0.995)
    fig.savefig(out, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["mixed", "large"])
    args = parser.parse_args()
    cfg = load_config()
    rep = Repairer(cfg)
    g = np.load(project_path(cfg["gallery"]))
    out_dir = project_path("outputs/interface")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "mixed"):
        mixed_figure(rep, g, out_dir / "hybrid_samples.png")
    if args.only in (None, "large"):
        large_figure(rep, g, out_dir / "hybrid_samples_large.png")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

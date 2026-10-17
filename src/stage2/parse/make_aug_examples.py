"""Figure: 12 augmented parser training crops with their labels (check before training).

    python -m src.stage2.parse.make_aug_examples --data <folder of *_img.png / *_lbl.png> \
        [--out outputs/stage2/parser_aug_examples.png] [--seed 0]

Each pair: the augmented greyscale crop exactly as the network sees it | its label map
(grey = 255 ignore). Crops come from ParserTrainSet's own sampling (scale, rare-class
centring, rotation/flip, photometric augmentations), taken in order with a fixed seed.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.stage2.data.classes16 import C16
from src.stage2.parse.dataset import list_plans, random_crop, read_pair
from src.stage2.parse.train import load_cfg
from src.utils import PROJECT_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage2" / "parser_v1.yaml"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "stage2" / "parser_aug_examples.png"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    cfg = load_cfg(args.config, smoke=False)
    rng = np.random.default_rng(args.seed)
    plans = list_plans(args.data)
    cols = 4
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(4.2 * cols, 2.3 * rows))
    for i in range(args.n):
        plan = plans[i % len(plans)]
        img, lbl = read_pair(args.data, plan)
        x, y = random_crop(img, lbl, rng, cfg["data"], cfg["augment"], cfg["data"]["crop"])
        a, b = axes[i // cols, 2 * (i % cols)], axes[i // cols, 2 * (i % cols) + 1]
        a.imshow(x, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_title(plan, fontsize=7)
        b.imshow(C16.colourise(y), interpolation="nearest")
        b.set_title("labels", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    handles = [Patch(color=C16.palette[c] / 255, label=n) for c, n in enumerate(C16.names)]
    handles.append(Patch(color=(0.5, 0.5, 0.5), label="ignore (255)"))
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

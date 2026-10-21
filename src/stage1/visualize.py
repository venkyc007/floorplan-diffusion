"""Turning plan arrays into coloured pictures."""

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.stage1.data.classes import CLASSES, COLORS, NUM_CLASSES


def colorize(label):
    """Label map (H, W) with values 0..5 -> RGB image (H, W, 3)."""
    palette = np.array([COLORS[c] for c in range(NUM_CLASSES)], dtype=np.uint8)
    return palette[label]


def save_grid(labels, titles, path, ncols=4):
    """Save several label maps side by side as one PNG, with a colour legend."""
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows + 0.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, label, title in zip(np.ravel(axes), labels, titles):
        ax.imshow(colorize(label), interpolation="nearest")
        ax.set_title(title, fontsize=9)
    legend = [Patch(facecolor=np.array(COLORS[c]) / 255, edgecolor="grey", label=CLASSES[c])
              for c in range(NUM_CLASSES)]
    # Narrow grids get a 2-row legend so it isn't cut off at the edges.
    fig.legend(handles=legend, loc="lower center", frameon=False,
               ncol=NUM_CLASSES if ncols >= 3 else 3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)

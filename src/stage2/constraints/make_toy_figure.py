"""Gate G1 figure: every broken toy plan next to its violation map.

    python -m src.stage2.constraints.make_toy_figure   -> outputs/stage2/constraint_toys.png

Top-left: the valid toy (all constraints = 0 on it). Then one pair per constraint
C1-C18 and rule R1, R3, R4: the broken plan (hole outlined, if the constraint uses
one) | the plan faded with the violation in red, and the total violation in the title.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.stage2.constraints import toys
from src.stage2.constraints.constraints import REGISTRY, Ctx, one_hot, violation_map
from src.stage2.constraints.rules import r1_room_area, r3_adjacency, r4_exterior_door
from src.stage2.data.classes16 import C16
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2" / "constraint_toys.png"
RULES = {"R1": (r1_room_area, "room area in range"), "R3": (r3_adjacency, "plausible door adjacency"),
         "R4": (r4_exterior_door, "an exterior door exists")}


def violation(name, lab, hole):
    if name in RULES:
        return RULES[name][0](lab, toys.TOY_CONST)[1].astype(float)
    P = one_hot(torch.from_numpy(lab)[None])
    if name in ("C1", "C8"):
        ctx = Ctx(const=toys.TOY_CONST, K=200)
    else:
        h = None if hole is None else torch.from_numpy(hole).float()[None, None]
        ctx = Ctx(hole=h, Y=one_hot(torch.from_numpy(toys.valid_plan())[None]), const=toys.TOY_CONST, K=200)
    return violation_map(name, P, ctx)[0, 0].numpy()


def overlay(lab, v):
    rgb = C16.colourise(lab).astype(float) / 255
    faded = 0.35 * rgb + 0.65
    a = np.clip(v, 0, 1)[..., None]
    return faded * (1 - a) + np.array([0.9, 0.0, 0.0]) * a


def main():
    names = list(REGISTRY) + list(RULES)
    cols = 3
    rows = (len(names) + 1 + cols - 1) // cols
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(4 * cols, 2.1 * rows))
    cells = [axes[r, c:c + 2] for r in range(rows) for c in range(0, 2 * cols, 2)]
    valid = toys.valid_plan()
    cells[0][0].imshow(C16.colourise(valid), interpolation="nearest")
    cells[0][0].set_title("valid toy", fontsize=8)
    total = sum(float(violation(n, valid, toys.broken(n)[1]).sum()) for n in names)
    cells[0][1].imshow(overlay(valid, np.zeros(valid.shape)), interpolation="nearest")
    cells[0][1].set_title(f"all violations: {total:.1f}", fontsize=8)
    for (a, b), name in zip(cells[1:], names):
        lab, hole = toys.broken(name)
        v = violation(name, lab, hole)
        a.imshow(C16.colourise(lab), interpolation="nearest")
        if hole is not None:
            a.contour(hole, levels=[0.5], colors="k", linewidths=0.8, linestyles="--")
        title = REGISTRY[name].title if name in REGISTRY else RULES[name][1]
        a.set_title(f"{name}: {title}", fontsize=7)
        b.imshow(overlay(lab, v), interpolation="nearest")
        b.set_title(f"violation sum {v.sum():.1f}", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

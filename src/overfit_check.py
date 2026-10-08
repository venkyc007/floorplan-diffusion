"""Overfit check: can each model memorise ONE batch of 8 plans?

If a model can't even learn 8 fixed examples, something is broken (wrong
target, wrong loss, wrong input). Runs on CPU in a few minutes with tiny
models (the `smoke:` sizes from the configs).

    python -m src.overfit_check [--steps 400]

Saves outputs/overfit_check.png (loss curves + predictions) and prints
whether each loss dropped enough.
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.corruptions import to_model_input
from src.eval.figures import colorize
from src.eval.metrics import load_benchmark, sample_stats, summarize
from src.models import build_model
from src.utils import load_config, project_path

PASS_RATIO = 0.5   # final loss must be below this fraction of the initial loss


def fixed_batch(n=8):
    """n val-benchmark cases from different plans, cycling through the 5 types."""
    bench, meta = load_benchmark(project_path("data/benchmarks/val_benchmark.npz"))
    idx = [k * 50 * 5 + (k % 5) for k in range(n)]
    cond = np.stack([to_model_input(bench["inputs"][i], bench["masks"][i]) for i in idx])
    target = np.stack([(np.arange(6)[:, None, None] == bench["targets"][i][None]) for i in idx])
    batch = (torch.from_numpy(cond), torch.from_numpy(target.astype(np.float32)),
             torch.from_numpy(bench["masks"][idx][:, None].astype(np.float32)))
    return batch, bench, idx, meta.loc[idx, "type"].tolist()


def overfit(cfg_path, batch, steps, lr):
    cfg = load_config(cfg_path)
    mcfg = {**cfg["model"], **cfg["smoke"]["model"]}
    torch.manual_seed(0)
    model = build_model(mcfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = model.loss(*batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    gen = torch.Generator().manual_seed(0)
    preds = model.predict(batch[0], steps=20, generator=gen).numpy()
    return mcfg["kind"], np.array(losses), preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    batch, bench, idx, types = fixed_batch()

    results = {}
    for cfg_path, lr in (("train_unet", 3e-3), ("train_diffusion", 3e-3)):
        t = time.time()
        kind, losses, preds = overfit(cfg_path, batch, args.steps, lr)
        first, last = losses[:20].mean(), losses[-20:].mean()
        scores = summarize([sample_stats(p, bench["targets"][i], bench["masks"][i])
                            for p, i in zip(preds, idx)])
        ok = last < PASS_RATIO * first
        print(f"{kind}: loss {first:.4f} -> {last:.4f} ({last / first:.0%} of start) "
              f"{'PASS' if ok else 'FAIL'} | on the batch: mIoU {scores['mIoU']:.3f}, "
              f"tolerant mIoU {scores['tmIoU']:.3f} | {time.time() - t:.0f}s")
        results[kind] = (losses, preds, ok)

    show = 4
    fig = plt.figure(figsize=(10, 3 + 2.4 * show))
    grid = fig.add_gridspec(show + 1, 4)
    for col, kind in enumerate(("unet", "diffusion")):
        ax = fig.add_subplot(grid[0, 2 * col:2 * col + 2])
        ax.plot(results[kind][0])
        ax.set(title=f"{kind}: training loss on one batch of 8", xlabel="step", yscale="log")
    for r in range(show):
        i = idx[r]
        panels = [(bench["inputs"][i], f"input ({types[r]})"), (results["unet"][1][r], "U-Net"),
                  (results["diffusion"][1][r], "diffusion (20 DDIM steps)"), (bench["targets"][i], "ground truth")]
        for c, (img, title) in enumerate(panels):
            ax = fig.add_subplot(grid[r + 1, c])
            # Show predictions composited with the known pixels, as evaluation does.
            if c in (1, 2):
                img = np.where(bench["masks"][i], img, bench["targets"][i])
            ax.imshow(colorize(img.astype(np.uint8)), interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(project_path("outputs/overfit_check.png"), dpi=90)
    if not all(r[2] for r in results.values()):
        raise SystemExit("overfit check FAILED")


if __name__ == "__main__":
    main()

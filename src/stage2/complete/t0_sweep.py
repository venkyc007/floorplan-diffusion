"""Is the hybrid better than the U-Net sketch alone? t0 sweep on the same 30 val holes as fast_vs_full, CPU.

    python -m src.stage2.complete.t0_sweep --zip completion_val.zip --rows unet fast200 fast300 fast400 fast600
    python -m src.stage2.complete.t0_sweep --zip completion_val.zip --rows full400     (adds a row)
    -> outputs/stage2/t0_sweep.md   (predictions cached in outputs/stage2/t0_sweep_preds.npz, not in git)

Rows: unet = the 16-class U-Net sketch alone; fastT = hybrid fast mode (N = 2, 15 DDIM steps from t0 = T,
guidance on the last 5); fullT = hybrid full mode (N = 4, the steps <= T of a 50-step schedule, guidance at
every t <= 600). The same t0 for every hole (no size bins), so the rows compare like with like.

Scores inside each hole, averaged per hole (classes present in prediction or truth):
  strict mIoU
  tolerant mIoU  wall / door / window / railing pixels match within 1 px (Stage 1 rule, src/eval/metrics.py)
  validity       % of holes with no more violations than the ground truth window (Stage 1 rule, 16 classes:
                 the verifier's thresholded counts, sampler.violation_breakdown)
  violations     mean per hole, and split by type: walls C1-C5 / openings C9-C14 + C18 / spaces C6-C8 + R1 + R3 /
                 fixtures C15-C16
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.stage2.complete import sampler
from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.windows import model_inputs
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import cfg, models
from src.utils import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "stage2"
CACHE = OUT / "t0_sweep_preds.npz"
THIN = [C16.index[c] for c in ("wall", "door", "window", "railing")]
GROUPS = {"walls": ["C1", "C2", "C3", "C4", "C5"],
          "openings": ["C9", "C10", "C11", "C12", "C13", "C14", "C18"],
          "spaces": ["C6", "C7", "C8", "R1", "R3"],
          "fixtures": ["C15", "C16"]}


def hole_ious(pred, gt, mask):
    """(strict mIoU, tolerant mIoU) inside one hole."""
    k = np.ones((3, 3), np.uint8)
    strict, tol = [], []
    for c in range(C16.n):
        p, g = (pred == c) & mask, (gt == c) & mask
        union = (p | g).sum()
        if not union:
            continue
        strict.append((p & g).sum() / union)
        if c in THIN:
            m = ((p & cv2.dilate(g.astype(np.uint8), k).astype(bool)).sum() +
                 (g & cv2.dilate(p.astype(np.uint8), k).astype(bool)).sum()) / 2
            tol.append(m / (p.sum() + g.sum() - m))
        else:
            tol.append(strict[-1])
    return float(np.mean(strict)), float(np.mean(tol))


def predict(row, m, cond, sketch, fc, fast, seed):
    if row == "unet":
        return sketch
    mode, t0 = row[:4], int(row[4:])
    kw = ({"n": fast["n_samples"], "steps_from_t0": fast["steps_from_t0"], "guide_last": fast["guide_last"]}
          if mode == "fast" else {"n": fc["n_samples"], "steps": fc["ddim_steps"]})
    out, _ = sampler.fill(m["diffusion"], cond, sketch=sketch, t0=t0, eta_guidance=fc["eta_guidance"],
                          generator=torch.Generator().manual_seed(seed), **kw)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--rows", nargs="+", default=["unet", "fast200", "fast300", "fast400", "fast600"])
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT / "t0_sweep.md"))
    args = ap.parse_args()
    cache_path = Path(args.cache)
    c = cfg()
    fc, fast = c["fill"], c["fast"]
    m = models()
    labels, masks, kinds, _ = fixed_windows(load_plans(args.zip), args.n, seed=7)   # = fast_vs_full holes
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    sketches = []
    for i in range(args.n):
        cond, _, _ = model_inputs(labels[i:i + 1], masks[i:i + 1])
        with torch.no_grad():
            sketches.append(m["unet"].predict(cond))
    for row in args.rows:
        if row in cache:
            continue
        preds = []
        for i in range(args.n):
            cond, _, _ = model_inputs(labels[i:i + 1], masks[i:i + 1])
            preds.append(predict(row, m, cond, sketches[i], fc, fast, i)[0].cpu().numpy().astype(np.uint8))
            print(row, f"hole {i + 1}/{args.n}", flush=True)
        cache[row] = np.stack(preds)
        np.savez_compressed(cache_path, **cache)

    # ground-truth violations per hole (for validity)
    cond, _, mask_t = model_inputs(labels, masks)
    gt_v = sum(sampler.violation_breakdown(torch.from_numpy(labels.astype(np.int64)), mask_t, cond[:, :C16.n]).values())
    order = ["unet"] + sorted((r for r in cache if r != "unet"), key=lambda r: (r[:4], int(r[4:])))
    L = ["# U-Net sketch alone vs hybrid at several t0 (CPU)", "",
         f"The same {args.n} val holes as fast_vs_full.md (fixed windows, seed 7; kinds: " +
         ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))) + "). Same t0 for every hole. "
         f"Ground-truth windows: {gt_v.mean():.2f} violations per hole on average. Means per hole.", "",
         "| row | strict mIoU | tolerant mIoU | validity % | violations / hole | walls | openings | spaces | fixtures |",
         "|---|---|---|---|---|---|---|---|---|"]
    for row in order:
        pred = cache[row]
        ious = np.array([hole_ious(pred[i], labels[i], masks[i]) for i in range(args.n)])
        bd = sampler.violation_breakdown(torch.from_numpy(pred.astype(np.int64)), mask_t, cond[:, :C16.n])
        tot = sum(bd.values())
        grp = [np.mean(sum(bd[k] for k in GROUPS[g])) for g in GROUPS]
        L.append(f"| {row} | {100 * ious[:, 0].mean():.1f} | {100 * ious[:, 1].mean():.1f} | "
                 f"{100 * np.mean(tot <= gt_v):.0f} | {tot.mean():.2f} | " + " | ".join(f"{g:.2f}" for g in grp) + " |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

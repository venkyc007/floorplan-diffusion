"""Fast (live-app) vs full (evaluation) FILL settings on the same val holes, CPU.

    python -m src.stage2.complete.fast_vs_full --zip completion_val.zip [--n 30] -> outputs/stage2/fast_vs_full.md

Holes: n fixed val windows around random typed deletions (seed 7). Both modes start from the same
16-class U-Net sketch and use the K1 constrained diffusion, t0 by hole share as in the pipeline,
guidance eta from configs/stage2/pipeline.yaml.
  full  N = 4, the steps <= t0 of a 50-step DDIM schedule, guidance at every step with t <= 600
  fast  N = 2, 15 DDIM steps from t0, guidance only on the last 5 steps
Scores inside the holes: pixel accuracy and pooled mIoU (gates34.hole_scores), the chosen sample's
violation count (verifier), and seconds per hole.
"""

import argparse
import time

import numpy as np
import torch

from src.stage2.complete import sampler
from src.stage2.complete.data16 import fixed_windows, load_plans
from src.stage2.complete.gates34 import hole_scores
from src.stage2.complete.windows import model_inputs
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import cfg, models
from src.utils import PROJECT_ROOT


def t0_for(label, hole, fc):
    building = max(1, int(np.isin(label, C16.group("BOUNDARY") + C16.group("S") + C16.group("F")).sum()))
    share = hole.sum() / building
    return fc["t0"][0] if share < fc["size_bins"][0] else (fc["t0"][1] if share < fc["size_bins"][1] else fc["t0"][2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    c = cfg()
    fc, fast = c["fill"], c["fast"]
    m = models()
    labels, masks, kinds, _ = fixed_windows(load_plans(args.zip), args.n, seed=7)
    res = {"full": {"pred": [], "viol": [], "sec": []}, "fast": {"pred": [], "viol": [], "sec": []}}
    for i in range(args.n):
        cond, target, mask = model_inputs(labels[i:i + 1], masks[i:i + 1])
        with torch.no_grad():
            sketch = m["unet"].predict(cond)
        t0 = t0_for(labels[i], masks[i], fc)
        for mode, kw in (("full", {"n": fc["n_samples"], "steps": fc["ddim_steps"]}),
                         ("fast", {"n": fast["n_samples"], "steps_from_t0": fast["steps_from_t0"],
                                   "guide_last": fast["guide_last"]})):
            g = torch.Generator().manual_seed(i)
            t = time.time()
            out, info = sampler.fill(m["diffusion"], cond, sketch=sketch, t0=t0, eta_guidance=fc["eta_guidance"],
                                     generator=g, **kw)
            res[mode]["sec"].append(time.time() - t)
            res[mode]["pred"].append(out)
            res[mode]["viol"].append(info[0]["violations"][info[0]["chosen"]])
        print(f"hole {i + 1}/{args.n} ({kinds[i]}, t0 {t0}): full {res['full']['sec'][-1]:.0f} s, "
              f"fast {res['fast']['sec'][-1]:.0f} s", flush=True)
    cond, target, mask = model_inputs(labels, masks)
    rows = []
    for mode in ("full", "fast"):
        acc, miou = hole_scores(torch.cat(res[mode]["pred"]), target, mask)
        rows.append((mode, acc, miou, np.mean(res[mode]["viol"]), np.mean(res[mode]["sec"]), np.median(res[mode]["sec"])))
    L = ["# Fast (live app) vs full (evaluation) FILL settings, CPU", "",
         f"{args.n} val holes (fixed windows, seed 7; kinds: " +
         ", ".join(f"{k} {kinds.count(k)}" for k in sorted(set(kinds))) + f"), guidance eta {fc['eta_guidance']}. "
         "full: N = 4, steps <= t0 of a 50-step schedule, guidance at every t <= 600; fast: N = 2, 15 steps from t0, "
         "guidance on the last 5 steps. Both from the same U-Net sketch.", "",
         "| mode | hole pixel acc | hole mIoU | violations (chosen sample) | s / hole (mean) | s / hole (median) |",
         "|---|---|---|---|---|---|"]
    L += [f"| {r[0]} | {100 * r[1]:.1f} | {100 * r[2]:.1f} | {r[3]:.2f} | {r[4]:.0f} | {r[5]:.0f} |" for r in rows]
    (PROJECT_ROOT / "outputs" / "stage2" / "fast_vs_full.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

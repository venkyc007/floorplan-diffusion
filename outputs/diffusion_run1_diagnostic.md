# Diffusion run 1: diagnostic

`best.pt` (EMA weights, step 26,924) of `venkatesh007c/floorplan-diffusion`, scored on the **same 256 fixed val cases** used for training validation. Run on Kaggle 2026-10-09 (kernel `venkatesh007c/floorplan-diagnostic`, T4 ×2, one half of the settings per GPU; val checksum matched). Script: `src/diagnose_diffusion.py`. Figure: `diffusion_run1_diagnostic.png`.

Metrics as defined in `docs/decisions.md`. Prediction outside the mask is replaced by the known pixels before scoring. Speckles = tiny pieces (< 4 px, any class, 4-connected) touching the mask, per case. Seconds per case on **one** T4 at batch 32 (the GPUs were throttled to about 705 MHz, see item 8).

## 1–2, 5, 6: Sampling sweep, one-shot, known-region reset, speckles

| Setting | Tolerant mIoU | mIoU | F1 door | F1 window | F1 room | Conn. error | Speckles / case | s / case |
|---|---|---|---|---|---|---|---|---|
| **U-Net baseline** (best.pt) | **0.793** | 0.728 | **0.653** | **0.705** | 0.942 | 0.133 | 0.24 | 0.018 |
| Diffusion one-shot, t = T, pure noise | 0.792 | **0.734** | 0.648 | 0.685 | 0.946 | 0.051 | 0.20 | 0.011 |
| Diffusion one-shot, t = T, clean known region filled in | 0.528 | 0.450 | 0.485 | 0.344 | 0.653 | 1.508 | 4.67 | 0.011 |
| DDIM 20, eta 0 (training validation) | 0.698 | 0.641 | 0.540 | 0.554 | 0.910 | 0.031 | 0.14 | 0.23 |
| DDIM 50, eta 0 | 0.728 | 0.670 | 0.585 | 0.598 | 0.939 | 0.004 | 0.11 | 0.52 |
| DDIM 100, eta 0 | 0.727 | 0.671 | 0.594 | 0.550 | 0.937 | 0.012 | 0.08 | 1.12 |
| DDIM 50, eta 1 (stochastic) | 0.753 | 0.697 | 0.590 | 0.594 | 0.946 | **0.000** | **0.04** | 0.57 |
| DDPM 250 | 0.753 | 0.696 | 0.629 | 0.610 | 0.941 | 0.004 | 0.05 | 2.77 |
| DDIM 50, eta 0, **no** known-region reset | 0.729 | 0.669 | 0.596 | 0.600 | **0.950** | 0.020 | 0.09 | 0.56 |
| Ground truth (reference) | 1 | 1 | 1 | 1 | 1 | 0 | 0.05 | — |

- **More steps help up to 50, then stop:** DDIM 20 → 50 → 100 gives 0.698 → 0.728 → 0.727.
- **Stochastic sampling helps:** eta 1 at 50 steps scores 0.753, the same as DDPM 250, at about 1/5 of the cost.
- **The one-shot prediction at t = T already matches the U-Net** (0.792 against 0.793). It's the same kind of output, a per-pixel best guess, and it has fewer speckles and a lower connectivity error.
- **Filling the clean known region into the noisy target at t = T breaks the model** (0.528, 4.7 speckles per case). The model only ever saw noisy targets at t = T, so a clean region there is unfamiliar input.
- **The known-region reset makes no difference to the scores** (0.728 on, 0.729 off). Without it the connectivity error is slightly higher (0.020 against 0.004). The network already receives the known region through the condition channels.
- **Speckles:** the U-Net (0.24) and the one-shot pass (0.20) leave the most; DDPM 250 (0.05) and DDIM eta 1 (0.04) match the ground truth's own 0.05. The connectivity error follows the same pattern: 0.133 for the U-Net against 0.000–0.004 for the stochastic samplers.

## 3: Denoiser quality per noise level

Ground truth noised to level t (whole target), one forward pass, argmax of the predicted x0.

| t | 50 | 200 | 400 | 600 | 800 | 999 (≈ pure noise) |
|---|---|---|---|---|---|---|
| Tolerant mIoU | 1.000 | 1.000 | 0.998 | 0.991 | 0.949 | 0.792 |

**The denoiser is very good.** It reconstructs the masked region almost perfectly from moderately noisy versions of the truth, and at full noise it matches the U-Net. (t = 999 and the pure-noise one-shot give identical numbers, as expected: the cosine schedule has almost no signal left at t = 999.)

## 4: Final output choice

**argmax of the final x_t and argmax of the last predicted x0 are identical in every metric, for every sampler.** With DDIM (`set_alpha_to_one=True`) and DDPM, the last step returns exactly the predicted x0, and outside the mask both get the known pixels. The code (`DiffusionInpainter.predict`) uses **argmax of the final x_t** after the last known-region reset.

## 7: Training samples (global batch × steps)

| Run | GPUs | Batch per step | Steps | Samples |
|---|---|---|---|---|
| U-Net baseline | 1 T4 | 32 | 8,000 | **256,000** |
| Diffusion run 1 | 2 T4 (DDP) | 64 (32 per GPU) | 30,490 | **1,951,360** |

Ratio **7.6×**, confirmed from both runs' `config.yaml` (batch 32) and logs (the U-Net ran single-process; diffusion's `train_log.csv` shows `gpus = 2`). An estimate of ~4× would assume the U-Net also used 64 per step.

## 8: GPU clock and temperature (now logged)

`gpu_usage.csv` now records SM clock, maximum SM clock, temperature and power. In this run, while the GPUs were busy:

- **SM clock: median 705 MHz** (range 600–1,125) against a **maximum of 1,590 MHz**;
- **temperature: median 77 °C** (maximum 80); power about 66 W (the T4 limit is 70 W).

The clocks fell from about 1,100 MHz to 600–765 MHz within about 2 minutes of full load. That's **thermal/power throttling**, the likely reason training ran at 66 instead of about 75 samples/s. It also means the seconds per case above are throttled speeds.

## Interpretation

- **The diffusion model isn't broken, and sampling isn't the main cause either.** Its denoiser is excellent, and its one-pass prediction equals the U-Net.
- **The gap comes from what the metrics reward.** Pixel IoU and F1 reward the **per-pixel best guess** (the U-Net and the one-shot pass). Diffusion **samples** commit to **one** complete, plausible layout, and for large holes that is often a different layout from the true one, which IoU penalises. The figure shows it: the U-Net and one-shot draw blurry, half-finished walls in large holes; the samplers draw clean, complete rooms that don't match the ground truth.
- **On structure, the samplers are clearly better:** connectivity error ≈ 0, speckles at the ground-truth level, and room F1 equal or higher.

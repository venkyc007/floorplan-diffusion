# Diffusion run 1: sampling experiment

`best.pt` (EMA, step 26,924) on the **same 256 fixed val cases** as the diagnostic (`diffusion_run1_diagnostic.md`). Kaggle 2026-10-09, kernel `venkatesh007c/floorplan-sampling`, T4 ×2 (one half of the work per GPU), val checksum matched. Script: `src/sampling_experiment.py`. Figure: `diffusion_run1_sampling.png` (it shows refine t = 400, the best refine by tolerant mIoU).

Seconds per case on one throttled T4 at batch 32. Speckles = pieces < 4 px touching the mask, per case (the ground truth has 0.05).

| Setting | Tolerant mIoU | mIoU | F1 door | F1 window | F1 room | Conn. error | Room leak | Room count error | Speckles | s / case |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net baseline | 0.793 | 0.728 | 0.653 | 0.705 | 0.942 | 0.133 | 0.052 | 0.238 | 0.24 | 0.017 |
| Diffusion one-shot (t = T) | 0.794 | 0.735 | 0.654 | 0.665 | 0.942 | 0.078 | 0.043 | 0.250 | 0.15 | 0.014 |
| DDIM 50, eta 1 (1 sample) | 0.744 | 0.689 | 0.616 | 0.558 | 0.950 | 0.000 | 0.002 | 0.148 | 0.05 | 0.49 |
| Refine from one-shot, t = 200 | 0.795 | 0.739 | 0.690 | 0.704 | 0.949 | 0.023 | 0.031 | 0.223 | 0.04 | 0.12 |
| Refine from one-shot, t = 400 | **0.796** | **0.740** | 0.689 | 0.704 | 0.949 | 0.023 | 0.028 | 0.219 | 0.03 | 0.22 |
| **Refine from one-shot, t = 600** | 0.794 | 0.739 | **0.708** | **0.718** | 0.951 | 0.016 | 0.014 | 0.180 | 0.04 | 0.32 |
| Medoid of 4 samples (no ground truth) | 0.776 | 0.721 | 0.668 | 0.638 | **0.956** | 0.004 | **0.002** | **0.144** | **0.03** | 2.09 |
| Pixel vote of 4 samples (re-blurs) | 0.775 | 0.717 | 0.623 | 0.622 | 0.949 | 0.062 | 0.025 | 0.164 | 0.20 | 2.09 |
| *ORACLE best of 4 vs ground truth (upper bound, needs the answer)* | *0.845* | *0.786* | *0.694* | *0.731* | *0.968* | *0.004* | *0.004* | *0.121* | *0.04* | *2.09* |

- **Refining from the one-shot guess gives the best of both.** It keeps the per-pixel accuracy of the one-shot/U-Net (tolerant mIoU 0.794–0.796) and adds the samplers' clean structure. Speckles drop to the ground-truth level, and the connectivity error falls from 0.133 (U-Net) to 0.016–0.023. **t = 600** is the best trade-off: the best door/window F1, room leak 0.014 (U-Net 0.052) and room count error 0.180 (U-Net 0.238), at 0.32 s per case. It's about 20× slower than the U-Net but still fast.
- **The medoid of 4** picks one sample without the ground truth. It has the cleanest structure of all the usable methods (connectivity error 0.004, room leak 0.002), but lower overlap (0.776) at 4× the cost. Its picks were spread across the 4 samples (88/60/58/50), so it isn't stuck on one.
- **The pixel vote of 4** brings the blur back: speckles 0.20 and connectivity error 0.062, almost as bad as the U-Net.
- **The oracle (0.845) is only an upper bound.** It shows that among 4 samples there is often one much closer to the truth than the average sample (0.744). Diffusion's diversity contains the answer; picking it without the ground truth is the hard part.
- **Seed variation:** the single DDIM 50 eta-1 sample scored 0.744 here and 0.753 in the diagnostic, so differences of about ±0.01 between single-sample settings are within the noise of the random sampling.

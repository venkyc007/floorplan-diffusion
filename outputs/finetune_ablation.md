# Fine-tune ablation: structural constraints vs control

Both runs start from diffusion run 1's `best.pt` (EMA weights, step 26,924) and are **identical except the constraint switch** (enforced by `tests/test_finetune_configs.py`): 26,000 steps × 64 (2 T4s × 32), learning rate 2e-5 with 200 warm-up steps and cosine decay, EMA 0.999, stochastic DDIM validation (20 steps) on the 256 fixed val cases. Kaggle 2026-10-09/10, kernels `venkatesh007c/floorplan-finetune-constraints` and `-control`, launched together at 19:55 IST. Logs: `finetune_runs/`. Figure: `finetune_ablation.png`.

Constraint run: **clDice 0.01 + ground-truth-relative closure 0.03** at all noise levels; openings and alignment off. These are the weights that passed the overfit test (`docs/decisions.md`); the first weights were about 100× too strong.

| Validation (256 cases) | Constraints | Control |
|---|---|---|
| Tolerant mIoU, first validation (≈2,000 steps) | 0.734 | 0.734 |
| Tolerant mIoU at 26,000 steps | **0.7512** | **0.7511** |
| Best tolerant mIoU (step) | 0.7513 (21,038) | 0.7513 (22,267) |
| mIoU at the end | 0.694 | 0.694 |
| F1 door / window / room at the end | 0.600 / 0.616 / 0.946 | 0.605 / 0.618 / 0.947 |
| Connectivity error at the end | 0.012 | 0.008 |
| Room leak at the end | 0.006 | 0.009 |
| Room count error at the end | 0.133 | 0.148 |
| Training time, speed | 6.81 h, 69.9 samples/s | 7.03 h, 67.6 samples/s |

- **No measurable effect.** The two runs track each other at every validation point. Differences in the structural metrics are small, go both ways during the run, and are within the noise of 256 cases. Full-val numbers: `full_eval.md` (pending).
- **The constraints were active:** the clDice term fell from about 0.0099 to 0.0068 during the constraint run. At a weight small enough not to damage reconstruction, it didn't change the sampled outputs measurably.
- **Why this is plausible:** the run-1 diagnostic found the diffusion **samples** already structurally clean (connectivity error ≈ 0, speckles at the ground-truth level). The structural weaknesses are in the U-Net and the one-shot prediction, which training-time losses on the sampler don't target.
- **The fine-tune itself added little** (0.734 → 0.751 in both runs). Most of the gain over run 1's training validation (0.698, deterministic DDIM 20) comes from the switch to stochastic sampling.
- **Speed varies between Kaggle machines:** the constraint run was faster despite the extra work (median SM clock 810 MHz against about 705 MHz the day before).

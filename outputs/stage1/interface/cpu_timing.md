# Repair time per method

One repair per method and hole size (seed 0, after a warm-up), on **CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel, 10 threads**, PyTorch 2.14.1+cpu. Diffusion cost grows with t₀: the steps at or below t₀ of the 50-step DDIM schedule are run (t₀ = 400 → 21 steps, 600 → 31, 700 → 36).

| method | small hole | medium hole | large hole |
|---|---|---|---|
| Size-adaptive hybrid (recommended) | 29.28 s (t₀ 400) | 44.16 s (t₀ 600) | 52.31 s (t₀ 700) |
| Hybrid, t₀ = 600 | 45.38 s (t₀ 600) | 45.82 s (t₀ 600) | 46.66 s (t₀ 600) |
| Matched U-Net | 1.40 s | 1.35 s | 1.36 s |
| Diffusion, refine-600 | 45.11 s (t₀ 600) | 45.36 s (t₀ 600) | 45.53 s (t₀ 600) |
| Nearest fill (baseline) | 0.01 s | 0.00 s | 0.00 s |

Loading a checkpoint (once, on the first repair that needs it): unet_matched 0.7 s, diffusion 0.6 s.

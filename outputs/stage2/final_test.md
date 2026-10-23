# Stage 2 final TEST evaluation (CubiCasa test split, run once)

Code version: final 7581cb2. Settings: configs/stage2/test_eval.yaml; sampling: fast (the validated settings).

| section | planned | done | skipped | errors |
|---|---|---|---|---|
| fill | 120 | 120 | 0 | 0 |
| detect | 180 | 180 | 0 | 0 |
| e2e | 120 | 120 | 0 | 0 |

## 1. FILL with the hole given

Inside the hole; the same U-Net sketch and seed for every arm of a case. 'final' exists for medium / large holes only (small holes ship the constrained arm, plus the sketch prior and type restriction tested end to end).

| size bin | n | arm | tol. mIoU | validity | violations per hole (GT) | fixed |
|---|---|---|---|---|---|---|
| small | 40 | U-Net sketch alone | 80.6 | 25% | 5.28 (2.40) | 82% |
| small | 40 | U-Net sketch + constrained diffusion | 82.1 | 65% | 2.73 (2.40) | 82% |
| small | 40 | U-Net sketch + control diffusion | 81.9 | 65% | 2.75 (2.40) | 85% |
| medium | 40 | U-Net sketch alone | 54.5 | 18% | 24.57 (4.58) | 90% |
| medium | 40 | U-Net sketch + constrained diffusion | 54.3 | 57% | 6.83 (4.58) | 90% |
| medium | 40 | U-Net sketch + control diffusion | 54.9 | 57% | 6.33 (4.58) | 90% |
| medium | 40 | shipped large-hole method (B4 + big window) | 55.0 | 57% | 5.85 (4.58) | 88% |
| large | 40 | U-Net sketch alone | 49.4 | 10% | 40.62 (7.00) | 95% |
| large | 40 | U-Net sketch + constrained diffusion | 49.8 | 32% | 10.95 (7.00) | 95% |
| large | 40 | U-Net sketch + control diffusion | 50.4 | 38% | 10.50 (7.00) | 92% |
| large | 40 | shipped large-hole method (B4 + big window) | 55.4 | 45% | 9.03 (7.00) | 98% |
| all | 120 | U-Net sketch alone | 61.5 | 18% | 23.49 (4.66) | 89% |
| all | 120 | U-Net sketch + constrained diffusion | 62.1 | 52% | 6.83 (4.66) | 89% |
| all | 120 | U-Net sketch + control diffusion | 62.4 | 53% | 6.53 (4.66) | 89% |
| all | 80 | shipped large-hole method (B4 + big window) | 55.2 | 51% | 7.44 (5.79) | 92% |

### Ablation: constrained vs control diffusion (paired bootstrap 95% CI of constrained - control)

| size bin | n | tol. mIoU constrained / control | diff [95% CI] | validity constrained / control | violations per hole constrained / control | diff [95% CI] | fixed constrained / control |
|---|---|---|---|---|---|---|---|
| small | 40 | 82.1 / 81.9 | +0.3 [-2.7, +3.5] | 65% / 65% | 2.73 / 2.75 | -0.03 [-0.25, +0.17] | 82% / 85% |
| medium | 40 | 54.3 / 54.9 | -0.6 [-2.6, +1.2] | 57% / 57% | 6.83 / 6.33 | +0.50 [-0.15, +1.20] | 90% / 90% |
| large | 40 | 49.8 / 50.4 | -0.6 [-2.3, +0.9] | 32% / 38% | 10.95 / 10.50 | +0.45 [-0.38, +1.27] | 95% / 92% |
| all | 120 | 62.1 / 62.4 | -0.3 [-1.6, +1.0] | 52% / 53% | 6.83 / 6.53 | +0.31 [-0.03, +0.67] | 89% / 89% |

## 2. CHECK (detection) on test drawings

False alarms per intact drawing: 0.80 (60 drawings). Omissions: 120, found 70% (right type 36%).

By type:

| type | n | found | right type |
|---|---|---|---|
| door (a) | 6 | 67% | 33% |
| door (b) | 7 | 29% | 29% |
| window | 5 | 40% | 40% |
| wall | 4 | 75% | 0% |
| railing | 3 | 0% | 0% |
| fixture | 10 | 50% | 40% |
| room | 5 | 100% | 100% |
| room (whole) | 37 | 81% | 38% |
| region | 43 | 77% | 33% |

By location:

| location | n | found | right type |
|---|---|---|---|
| interior | 56 | 59% | 38% |
| exterior | 64 | 80% | 34% |

By position:

| position | n | found | right type |
|---|---|---|---|
| centre | 85 | 74% | 40% |
| edge | 35 | 60% | 26% |

By size_bin:

| size_bin | n | found | right type |
|---|---|---|---|
| small | 40 | 52% | 38% |
| medium | 40 | 72% | 32% |
| large | 40 | 85% | 38% |

## 3. End to end on test drawings

120 drawings: found 66%, right type 29%, fixed 12% (by addition 7%), false additions 0.85 per drawing, tolerant mIoU 44.4 -> 44.5, validity 7% -> 8%, 2 s per drawing.

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (a) | 3 | 67% | 33% | 0.33 | 64.0 |
| door (b) | 3 | 100% | 100% | 1.00 | 68.1 |
| window | 9 | 56% | 22% | 0.33 | 60.6 |
| wall | 7 | 57% | 29% | 1.00 | 47.1 |
| railing | 5 | 0% | 0% | 0.00 | 56.8 |
| fixture | 11 | 36% | 0% | 1.09 | 47.6 |
| room | 2 | 100% | 50% | 0.50 | 57.8 |
| room (whole) | 27 | 89% | 7% | 1.37 | 37.4 |
| region | 53 | 66% | 8% | 0.72 | 40.3 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 51 | 57% | 14% | 0.84 | 39.7 |
| exterior | 69 | 72% | 12% | 0.86 | 48.1 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 84 | 71% | 14% | 0.85 | 43.3 |
| edge | 36 | 53% | 8% | 0.86 | 47.3 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 40 | 50% | 22% | 0.68 | 54.9 |
| medium | 40 | 68% | 15% | 1.05 | 38.6 |
| large | 40 | 80% | 0% | 0.82 | 40.0 |

Size mix realised (detection omissions + end to end): {'small': 80, 'medium': 80, 'large': 80}.

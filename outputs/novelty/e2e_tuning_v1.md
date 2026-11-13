# End to end on drawings: shipped pipeline vs hybrid pipeline_tuning_v1

100 val drawings (TUNING drawings: used to choose settings, so optimistic). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 100 | 67% | 2% | **52%** [42, 62] | **53%** [30, 75] | 0.68 | 0.77 | 60.1 -> 63.0 | 9% -> 11% | 10.3 (3.8) | 0.84 | 3.8 |
| hybrid auto | 100 | 68% | 10% | **50%** [40, 60] | **53%** [30, 75] | 0.80 | 0.75 | 60.1 -> 62.3 | 9% -> 10% | 10.5 (3.8) | 1.70 | 5.1 |
| hybrid auto + outline | 100 | 68% | 10% | **49%** [39, 59] | **40%** [20, 64] | 0.80 | 0.61 | 60.1 -> 61.9 | 9% -> 10% | 11.0 (3.8) | 1.64 | 5.0 |

hybrid auto minus shipped auto (paired, 100 cases): fixed, points -2.0 [-5.0, +0.0]; tol. mIoU -0.7 [-1.6, +0.1]; valid, points -1.0 [-3.0, +0.0].

hybrid auto + outline minus hybrid auto (paired, 100 cases): fixed, points -1.0 [-3.0, +0.0]; tol. mIoU -0.4 [-0.8, -0.0]; valid, points +0.0 [+0.0, +0.0].

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 100 | 66% | 3% | **13%** [8, 21] | **12%** [7, 20] | 0.15 | 0.76 | 37.1 -> 38.8 | 3% -> 6% | 22.0 (7.1) | 1.01 | 3.7 |
| hybrid auto | 100 | 83% | 67% | **31%** [23, 41] | **41%** [32, 51] | 0.42 | 0.74 | 37.1 -> 41.6 | 3% -> 15% | 15.8 (7.1) | 2.52 | 7.0 |
| hybrid auto + outline | 100 | 83% | 67% | **33%** [25, 43] | **33%** [25, 43] | 0.53 | 0.54 | 37.1 -> 38.5 | 3% -> 3% | 22.6 (7.1) | 2.57 | 7.0 |
| shipped manual | 100 | 100% | 0% | **46%** [37, 56] | **48%** [38, 58] | 0.57 | 0.70 | 37.1 -> 41.1 | 3% -> 12% | 18.0 (7.1) | 0.28 | 1.4 |
| hybrid manual | 100 | 100% | 0% | **53%** [43, 62] | **64%** [54, 73] | 0.69 | 0.75 | 37.1 -> 45.3 | 3% -> 12% | 18.4 (7.1) | 0.40 | 4.8 |

hybrid auto minus shipped auto (paired, 100 cases): fixed, points +18.0 [+9.0, +27.0]; recovered, points +29.0 [+20.0, +38.0]; tol. mIoU +2.8 [-0.1, +5.6]; valid, points +9.0 [+3.0, +15.0].

hybrid manual minus shipped manual (paired, 100 cases): fixed, points +7.0 [+0.0, +14.0]; recovered, points +16.0 [+7.0, +25.0]; tol. mIoU +4.2 [+2.2, +6.2]; valid, points +0.0 [-5.0, +4.0].

hybrid auto + outline minus hybrid auto (paired, 100 cases): fixed, points +2.0 [-3.0, +7.0]; recovered, points -8.0 [-17.0, +0.0]; tol. mIoU -3.1 [-4.6, -1.8]; valid, points -12.0 [-19.0, -6.0].

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 100 | 77% | 1% | **7%** [3, 14] | **9%** [5, 16] | 0.13 | 0.71 | 36.4 -> 36.5 | 6% -> 6% | 31.0 (12.0) | 0.55 | 3.6 |
| hybrid auto | 100 | 99% | 85% | **29%** [21, 39] | **42%** [33, 52] | 0.47 | 0.66 | 36.4 -> 40.6 | 6% -> 25% | 21.2 (12.0) | 2.18 | 7.6 |
| hybrid auto + outline | 100 | 99% | 85% | **36%** [27, 46] | **24%** [17, 33] | 0.57 | 0.41 | 36.4 -> 36.0 | 6% -> 5% | 32.9 (12.0) | 1.97 | 7.6 |
| shipped manual | 100 | 100% | 0% | **30%** [22, 40] | **32%** [24, 42] | 0.48 | 0.57 | 36.4 -> 36.6 | 6% -> 22% | 22.2 (12.0) | 0.60 | 1.6 |
| hybrid manual | 100 | 100% | 0% | **32%** [24, 42] | **45%** [36, 55] | 0.55 | 0.67 | 36.4 -> 40.8 | 6% -> 17% | 23.6 (12.0) | 0.68 | 4.8 |

hybrid auto minus shipped auto (paired, 100 cases): fixed, points +22.0 [+14.0, +31.0]; recovered, points +33.0 [+24.0, +42.0]; tol. mIoU +4.1 [+1.9, +6.3]; valid, points +19.0 [+11.0, +27.0].

hybrid manual minus shipped manual (paired, 100 cases): fixed, points +2.0 [-5.0, +10.0]; recovered, points +13.0 [+6.0, +21.0]; tol. mIoU +4.2 [+2.6, +5.8]; valid, points -5.0 [-12.0, +2.0].

hybrid auto + outline minus hybrid auto (paired, 100 cases): fixed, points +7.0 [+2.0, +13.0]; recovered, points -18.0 [-28.0, -8.0]; tol. mIoU -4.6 [-5.9, -3.4]; valid, points -20.0 [-28.0, -12.0].

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 200 | 72% | 2% | **10%** [7, 15] | **10%** [7, 16] | 0.14 | 0.74 | 36.7 -> 37.7 | 4% -> 6% | 26.5 (9.6) | 0.78 | 3.7 |
| hybrid auto | 200 | 91% | 76% | **30%** [24, 37] | **42%** [35, 48] | 0.45 | 0.70 | 36.7 -> 41.1 | 4% -> 20% | 18.5 (9.6) | 2.35 | 7.3 |
| hybrid auto + outline | 200 | 91% | 76% | **34%** [28, 41] | **28%** [23, 35] | 0.55 | 0.47 | 36.7 -> 37.2 | 4% -> 4% | 27.7 (9.6) | 2.27 | 7.3 |
| shipped manual | 200 | 100% | 0% | **38%** [32, 45] | **40%** [33, 47] | 0.52 | 0.64 | 36.7 -> 38.9 | 4% -> 17% | 20.1 (9.6) | 0.44 | 1.5 |
| hybrid manual | 200 | 100% | 0% | **42%** [36, 49] | **54%** [48, 61] | 0.62 | 0.71 | 36.7 -> 43.1 | 4% -> 14% | 21.0 (9.6) | 0.54 | 4.8 |

hybrid auto minus shipped auto (paired, 200 cases): fixed, points +20.0 [+13.5, +26.0]; recovered, points +31.0 [+24.5, +38.0]; tol. mIoU +3.4 [+1.6, +5.3]; valid, points +14.0 [+9.0, +19.0].

hybrid manual minus shipped manual (paired, 200 cases): fixed, points +4.5 [-0.5, +9.5]; recovered, points +14.5 [+9.0, +20.5]; tol. mIoU +4.2 [+2.9, +5.5]; valid, points -2.5 [-6.5, +1.5].

hybrid auto + outline minus hybrid auto (paired, 200 cases): fixed, points +4.5 [+1.0, +8.5]; recovered, points -13.0 [-19.5, -6.5]; tol. mIoU -3.9 [-4.8, -2.9]; valid, points -16.0 [-21.0, -11.0].

## By omission kind (fixed)

| kind | n | shipped auto | hybrid auto | hybrid auto + outline | shipped manual | hybrid manual |
|---|---|---|---|---|---|---|
| wall | 17 | 47% | 41% | 35% | - | - |
| room (whole) | 82 | 10% | 29% | 38% | 24% | 30% |
| region | 118 | 10% | 31% | 32% | 47% | 51% |
| window | 20 | 75% | 75% | 75% | - | - |
| door (a) | 11 | 45% | 45% | 45% | - | - |
| room | 15 | 87% | 80% | 80% | - | - |
| fixture | 12 | 25% | 25% | 25% | - | - |
| door (b) | 14 | 57% | 57% | 57% | - | - |
| railing | 11 | 0% | 0% | 0% | - | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| shipped auto | 100 | 0.87 | 0.01 | 1% | 0.92 |
| hybrid auto | 100 | 1.03 | 0.20 | 16% | 1.57 |
| hybrid auto + outline | 100 | 1.03 | 0.20 | 16% | 1.52 |

# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_v1

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 61% | 1% | **42%** [34, 50] | **50%** [31, 69] | 0.63 | 0.77 | 63.0 -> 65.6 | 12% -> 13% | 9.2 (3.4) | 0.74 | 3.1 |
| hybrid auto | 151 | 64% | 8% | **42%** [35, 50] | **55%** [35, 73] | 0.68 | 0.77 | 63.0 -> 64.8 | 12% -> 13% | 9.1 (3.4) | 1.45 | 4.0 |

hybrid auto minus shipped auto (paired, 151 cases): fixed, points +0.7 [-1.3, +2.6]; tol. mIoU -0.7 [-1.4, -0.2]; valid, points +0.0 [+0.0, +0.0].

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 73% | 1% | **15%** [10, 21] | **17%** [12, 24] | 0.21 | 0.79 | 41.1 -> 42.6 | 4% -> 7% | 19.9 (6.5) | 0.69 | 3.3 |
| hybrid auto | 151 | 87% | 70% | **39%** [32, 47] | **50%** [42, 58] | 0.52 | 0.77 | 41.1 -> 44.0 | 4% -> 15% | 14.9 (6.5) | 2.09 | 6.2 |
| locator + B4 | 151 | 87% | 70% | **38%** [30, 46] | **45%** [37, 53] | 0.50 | 0.74 | 41.1 -> 43.4 | 4% -> 15% | 13.8 (6.5) | 1.97 | 3.8 |
| shipped manual | 151 | 100% | 0% | **47%** [39, 55] | **58%** [50, 65] | 0.63 | 0.77 | 41.1 -> 46.0 | 4% -> 9% | 15.3 (6.5) | 0.22 | 1.3 |
| hybrid manual | 151 | 100% | 0% | **50%** [42, 58] | **68%** [60, 75] | 0.69 | 0.80 | 41.1 -> 47.9 | 4% -> 6% | 16.6 (6.5) | 0.25 | 4.6 |

hybrid auto minus shipped auto (paired, 151 cases): fixed, points +24.5 [+16.6, +33.1]; recovered, points +33.1 [+25.1, +41.7]; tol. mIoU +1.5 [-0.6, +3.7]; valid, points +7.9 [+2.6, +13.9].

hybrid manual minus shipped manual (paired, 151 cases): fixed, points +3.3 [-2.0, +9.3]; recovered, points +9.9 [+2.6, +16.6]; tol. mIoU +1.9 [+0.5, +3.2]; valid, points -3.3 [-7.3, +0.7].

hybrid auto minus locator + B4 (paired, 151 cases): fixed, points +1.3 [-2.6, +5.3]; recovered, points +5.3 [+0.7, +10.6]; tol. mIoU +0.7 [-0.5, +1.8]; valid, points -0.7 [-6.0, +4.0].

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 80% | 3% | **13%** [8, 19] | **11%** [7, 17] | 0.13 | 0.70 | 37.4 -> 38.6 | 5% -> 7% | 29.6 (10.2) | 0.89 | 3.2 |
| hybrid auto | 151 | 95% | 80% | **32%** [25, 40] | **42%** [35, 50] | 0.46 | 0.67 | 37.4 -> 42.3 | 5% -> 15% | 21.4 (10.2) | 1.89 | 6.5 |
| locator + B4 | 151 | 95% | 80% | **30%** [24, 38] | **41%** [34, 49] | 0.46 | 0.63 | 37.4 -> 39.9 | 5% -> 17% | 20.1 (10.2) | 1.87 | 3.8 |
| shipped manual | 151 | 100% | 0% | **35%** [28, 43] | **35%** [28, 43] | 0.45 | 0.56 | 37.4 -> 39.4 | 5% -> 13% | 21.1 (10.2) | 0.75 | 1.5 |
| hybrid manual | 151 | 100% | 0% | **36%** [29, 44] | **50%** [42, 58] | 0.54 | 0.69 | 37.4 -> 43.4 | 5% -> 10% | 22.9 (10.2) | 0.72 | 4.6 |

hybrid auto minus shipped auto (paired, 151 cases): fixed, points +19.2 [+12.6, +26.5]; recovered, points +31.8 [+23.8, +39.1]; tol. mIoU +3.6 [+1.5, +5.7]; valid, points +7.9 [+2.6, +13.9].

hybrid manual minus shipped manual (paired, 151 cases): fixed, points +0.7 [-5.3, +6.6]; recovered, points +15.2 [+8.6, +21.9]; tol. mIoU +4.0 [+2.6, +5.4]; valid, points -3.3 [-6.6, +0.0].

hybrid auto minus locator + B4 (paired, 151 cases): fixed, points +1.3 [-2.6, +5.3]; recovered, points +1.3 [-3.3, +6.0]; tol. mIoU +2.4 [+1.1, +3.8]; valid, points -2.0 [-6.0, +2.0].

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 302 | 76% | 2% | **14%** [10, 18] | **14%** [10, 18] | 0.17 | 0.75 | 39.3 -> 40.6 | 5% -> 7% | 24.7 (8.4) | 0.79 | 3.3 |
| hybrid auto | 302 | 91% | 75% | **35%** [30, 41] | **46%** [41, 52] | 0.49 | 0.72 | 39.3 -> 43.2 | 5% -> 15% | 18.2 (8.4) | 1.99 | 6.4 |
| locator + B4 | 302 | 91% | 75% | **34%** [29, 40] | **43%** [38, 49] | 0.48 | 0.68 | 39.3 -> 41.6 | 5% -> 16% | 17.0 (8.4) | 1.92 | 3.8 |
| shipped manual | 302 | 100% | 0% | **41%** [36, 47] | **46%** [41, 52] | 0.54 | 0.66 | 39.3 -> 42.7 | 5% -> 11% | 18.2 (8.4) | 0.49 | 1.4 |
| hybrid manual | 302 | 100% | 0% | **43%** [38, 49] | **59%** [53, 64] | 0.62 | 0.74 | 39.3 -> 45.6 | 5% -> 8% | 19.8 (8.4) | 0.49 | 4.6 |

hybrid auto minus shipped auto (paired, 302 cases): fixed, points +21.9 [+16.9, +26.8]; recovered, points +32.5 [+27.2, +38.1]; tol. mIoU +2.6 [+1.2, +4.0]; valid, points +7.9 [+4.0, +12.3].

hybrid manual minus shipped manual (paired, 302 cases): fixed, points +2.0 [-2.0, +6.3]; recovered, points +12.6 [+7.9, +17.2]; tol. mIoU +3.0 [+1.9, +4.0]; valid, points -3.3 [-6.0, -0.7].

hybrid auto minus locator + B4 (paired, 302 cases): fixed, points +1.3 [-1.3, +4.3]; recovered, points +3.3 [+0.0, +6.6]; tol. mIoU +1.5 [+0.6, +2.4]; valid, points -1.3 [-4.6, +2.0].

## By omission kind (fixed)

| kind | n | shipped auto | hybrid auto | locator + B4 | shipped manual | hybrid manual |
|---|---|---|---|---|---|---|
| wall | 24 | 54% | 58% | - | - | - |
| room (whole) | 115 | 11% | 30% | 27% | 26% | 33% |
| region | 187 | 15% | 39% | 39% | 50% | 49% |
| window | 26 | 65% | 65% | - | - | - |
| door (a) | 18 | 28% | 28% | - | - | - |
| room | 22 | 73% | 73% | - | - | - |
| railing | 19 | 0% | 0% | - | - | - |
| door (b) | 23 | 48% | 48% | - | - | - |
| fixture | 19 | 5% | 5% | - | - | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| shipped auto | 151 | 0.64 | 0.01 | 1% | 0.68 |
| hybrid auto | 151 | 0.74 | 0.15 | 13% | 1.38 |

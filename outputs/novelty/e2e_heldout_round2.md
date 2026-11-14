# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_round2

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 61% | 1% | **42%** [34, 50] | **50%** [31, 69] | 0.63 | 0.77 | 63.0 -> 65.6 | 12% -> 13% | 9.2 (3.4) | 0.74 | 3.1 |
| hybrid v1 auto | 151 | 64% | 8% | **42%** [35, 50] | **55%** [35, 73] | 0.68 | 0.77 | 63.0 -> 64.8 | 12% -> 13% | 9.1 (3.4) | 1.45 | 4.0 |
| hybrid v2 auto | 151 | 64% | 8% | **42%** [34, 50] | **55%** [35, 73] | 0.64 | 0.80 | 63.0 -> 65.5 | 12% -> 12% | 9.2 (3.4) | 1.72 | 3.7 |
| hybrid v3 auto | 151 | 62% | 5% | **42%** [34, 50] | **55%** [35, 73] | 0.69 | 0.75 | 63.0 -> 65.4 | 12% -> 11% | 9.3 (3.4) | 1.16 | 3.6 |

hybrid v2 auto minus shipped auto (paired, 151 cases): fixed, points +0.0 [-2.0, +2.0]; tol. mIoU -0.0 [-0.4, +0.3]; valid, points -0.7 [-2.0, +0.0].

hybrid v1 auto minus shipped auto (paired, 151 cases): fixed, points +0.7 [-1.3, +2.6]; tol. mIoU -0.7 [-1.4, -0.2]; valid, points +0.0 [+0.0, +0.0].

hybrid v2 auto minus hybrid v1 auto (paired, 151 cases): fixed, points -0.7 [-2.0, +0.0]; tol. mIoU +0.7 [+0.2, +1.3]; valid, points -0.7 [-2.0, +0.0].

hybrid v3 auto minus shipped auto (paired, 151 cases): fixed, points +0.0 [-2.0, +2.0]; tol. mIoU -0.2 [-0.5, +0.1]; valid, points -1.3 [-3.3, +0.0].

hybrid v3 auto minus hybrid v2 auto (paired, 151 cases): fixed, points +0.0 [-2.6, +2.6]; tol. mIoU -0.1 [-0.7, +0.4]; valid, points -0.7 [-2.0, +0.0].

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 73% | 1% | **15%** [10, 21] | **17%** [12, 24] | 0.21 | 0.79 | 41.1 -> 42.6 | 4% -> 7% | 19.9 (6.5) | 0.69 | 3.3 |
| locator + B4 | 151 | 87% | 70% | **38%** [30, 46] | **45%** [37, 53] | 0.50 | 0.74 | 41.1 -> 43.4 | 4% -> 15% | 13.8 (6.5) | 1.97 | 3.8 |
| hybrid v1 auto | 151 | 87% | 70% | **39%** [32, 47] | **50%** [42, 58] | 0.52 | 0.77 | 41.1 -> 44.0 | 4% -> 15% | 14.9 (6.5) | 2.09 | 6.2 |
| hybrid v2 auto | 151 | 87% | 70% | **38%** [30, 46] | **50%** [42, 58] | 0.52 | 0.83 | 41.1 -> 47.3 | 4% -> 9% | 15.3 (6.5) | 2.33 | 5.9 |
| hybrid v3 auto | 151 | 91% | 79% | **54%** [46, 62] | **72%** [64, 78] | 0.75 | 0.79 | 41.1 -> 45.9 | 4% -> 11% | 14.4 (6.5) | 1.76 | 6.5 |
| shipped manual | 151 | 100% | 0% | **47%** [39, 55] | **58%** [50, 65] | 0.63 | 0.77 | 41.1 -> 46.0 | 4% -> 9% | 15.3 (6.5) | 0.22 | 1.3 |
| hybrid v1 manual | 151 | 100% | 0% | **50%** [42, 58] | **68%** [60, 75] | 0.69 | 0.80 | 41.1 -> 47.9 | 4% -> 6% | 16.6 (6.5) | 0.25 | 4.6 |
| hybrid v2 manual | 151 | 100% | 0% | **48%** [40, 56] | **70%** [62, 76] | 0.69 | 0.85 | 41.1 -> 50.2 | 4% -> 7% | 16.6 (6.5) | 0.25 | 4.5 |
| hybrid v3 manual | 151 | 100% | 0% | **69%** [61, 76] | **84%** [77, 89] | 0.89 | 0.79 | 41.1 -> 48.2 | 4% -> 3% | 18.8 (6.5) | 0.44 | 4.5 |

hybrid v2 auto minus shipped auto (paired, 151 cases): fixed, points +23.2 [+15.9, +31.1]; recovered, points +33.1 [+25.2, +41.1]; tol. mIoU +4.8 [+3.4, +6.2]; valid, points +2.6 [-2.0, +7.3].

hybrid v1 auto minus shipped auto (paired, 151 cases): fixed, points +24.5 [+16.6, +33.1]; recovered, points +33.1 [+25.1, +41.7]; tol. mIoU +1.5 [-0.6, +3.7]; valid, points +7.9 [+2.6, +13.9].

hybrid v2 auto minus locator + B4 (paired, 151 cases): fixed, points +0.0 [-6.6, +6.6]; recovered, points +5.3 [-1.3, +11.3]; tol. mIoU +3.9 [+2.2, +5.7]; valid, points -6.0 [-11.3, -0.7].

hybrid v2 auto minus hybrid v1 auto (paired, 151 cases): fixed, points -1.3 [-7.9, +5.3]; recovered, points +0.0 [-6.6, +6.6]; tol. mIoU +3.3 [+1.7, +4.8]; valid, points -5.3 [-10.6, +0.0].

hybrid v2 manual minus shipped manual (paired, 151 cases): fixed, points +0.7 [-7.3, +7.9]; recovered, points +11.9 [+4.0, +19.2]; tol. mIoU +4.2 [+2.5, +5.9]; valid, points -2.6 [-7.3, +1.3].

hybrid v2 manual minus hybrid v1 manual (paired, 151 cases): fixed, points -2.6 [-8.6, +3.3]; recovered, points +2.0 [-4.6, +7.9]; tol. mIoU +2.3 [+0.9, +3.7]; valid, points +0.7 [-2.6, +4.0].

hybrid v3 auto minus shipped auto (paired, 151 cases): fixed, points +39.7 [+31.8, +47.7]; recovered, points +54.3 [+46.4, +62.3]; tol. mIoU +3.3 [+1.7, +5.1]; valid, points +4.0 [-1.3, +8.6].

hybrid v3 auto minus hybrid v2 auto (paired, 151 cases): fixed, points +16.6 [+9.3, +24.5]; recovered, points +21.2 [+12.6, +29.8]; tol. mIoU -1.4 [-3.0, +0.1]; valid, points +1.3 [-4.0, +6.6].

hybrid v3 manual minus shipped manual (paired, 151 cases): fixed, points +21.9 [+13.9, +29.1]; recovered, points +26.5 [+17.9, +34.4]; tol. mIoU +2.2 [+0.7, +3.7]; valid, points -6.0 [-10.6, -1.3].

hybrid v3 manual minus hybrid v2 manual (paired, 151 cases): fixed, points +21.2 [+13.2, +28.5]; recovered, points +14.6 [+7.9, +21.2]; tol. mIoU -2.0 [-3.1, -1.0]; valid, points -3.3 [-7.3, +0.0].

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 80% | 3% | **13%** [8, 19] | **11%** [7, 17] | 0.13 | 0.70 | 37.4 -> 38.6 | 5% -> 7% | 29.6 (10.2) | 0.89 | 3.2 |
| locator + B4 | 151 | 95% | 80% | **30%** [24, 38] | **41%** [34, 49] | 0.46 | 0.63 | 37.4 -> 39.9 | 5% -> 17% | 20.1 (10.2) | 1.87 | 3.8 |
| hybrid v1 auto | 151 | 95% | 80% | **32%** [25, 40] | **42%** [35, 50] | 0.46 | 0.67 | 37.4 -> 42.3 | 5% -> 15% | 21.4 (10.2) | 1.89 | 6.5 |
| hybrid v2 auto | 151 | 95% | 80% | **36%** [29, 44] | **52%** [44, 59] | 0.51 | 0.80 | 37.4 -> 45.3 | 5% -> 13% | 22.5 (10.2) | 2.23 | 6.2 |
| hybrid v3 auto | 151 | 96% | 91% | **48%** [40, 56] | **75%** [67, 81] | 0.79 | 0.74 | 37.4 -> 44.0 | 5% -> 13% | 22.0 (10.2) | 2.84 | 7.2 |
| shipped manual | 151 | 100% | 0% | **35%** [28, 43] | **35%** [28, 43] | 0.45 | 0.56 | 37.4 -> 39.4 | 5% -> 13% | 21.1 (10.2) | 0.75 | 1.5 |
| hybrid v1 manual | 151 | 100% | 0% | **36%** [29, 44] | **50%** [42, 58] | 0.54 | 0.69 | 37.4 -> 43.4 | 5% -> 10% | 22.9 (10.2) | 0.72 | 4.6 |
| hybrid v2 manual | 151 | 100% | 0% | **42%** [34, 50] | **62%** [54, 70] | 0.60 | 0.82 | 37.4 -> 46.6 | 5% -> 11% | 23.5 (10.2) | 0.64 | 4.5 |
| hybrid v3 manual | 151 | 100% | 0% | **56%** [48, 63] | **79%** [72, 85] | 0.86 | 0.73 | 37.4 -> 45.2 | 5% -> 5% | 28.1 (10.2) | 1.21 | 4.5 |

hybrid v2 auto minus shipped auto (paired, 151 cases): fixed, points +23.2 [+15.9, +30.5]; recovered, points +41.1 [+32.5, +49.0]; tol. mIoU +6.7 [+4.9, +8.5]; valid, points +6.0 [+1.3, +10.6].

hybrid v1 auto minus shipped auto (paired, 151 cases): fixed, points +19.2 [+12.6, +26.5]; recovered, points +31.8 [+23.8, +39.1]; tol. mIoU +3.6 [+1.5, +5.7]; valid, points +7.9 [+2.6, +13.9].

hybrid v2 auto minus locator + B4 (paired, 151 cases): fixed, points +5.3 [-0.7, +11.3]; recovered, points +10.6 [+3.3, +17.2]; tol. mIoU +5.4 [+4.0, +6.8]; valid, points -4.0 [-9.3, +0.7].

hybrid v2 auto minus hybrid v1 auto (paired, 151 cases): fixed, points +4.0 [-1.3, +9.3]; recovered, points +9.3 [+3.3, +15.2]; tol. mIoU +3.0 [+1.7, +4.3]; valid, points -2.0 [-6.6, +2.6].

hybrid v2 manual minus shipped manual (paired, 151 cases): fixed, points +6.6 [-1.3, +13.9]; recovered, points +27.2 [+18.5, +35.1]; tol. mIoU +7.2 [+5.6, +8.8]; valid, points -2.6 [-6.6, +0.7].

hybrid v2 manual minus hybrid v1 manual (paired, 151 cases): fixed, points +6.0 [-1.3, +12.6]; recovered, points +11.9 [+4.0, +19.9]; tol. mIoU +3.2 [+1.9, +4.5]; valid, points +0.7 [-2.6, +4.0].

hybrid v3 auto minus shipped auto (paired, 151 cases): fixed, points +35.1 [+27.2, +43.0]; recovered, points +64.2 [+55.6, +72.2]; tol. mIoU +5.4 [+3.7, +7.0]; valid, points +6.0 [+1.3, +11.3].

hybrid v3 auto minus hybrid v2 auto (paired, 151 cases): fixed, points +11.9 [+5.3, +19.2]; recovered, points +23.2 [+15.9, +31.1]; tol. mIoU -1.3 [-2.6, +0.1]; valid, points +0.0 [-5.3, +5.3].

hybrid v3 manual minus shipped manual (paired, 151 cases): fixed, points +20.5 [+13.2, +28.5]; recovered, points +43.7 [+35.1, +51.7]; tol. mIoU +5.8 [+4.3, +7.2]; valid, points -7.9 [-12.6, -4.0].

hybrid v3 manual minus hybrid v2 manual (paired, 151 cases): fixed, points +13.9 [+7.3, +20.5]; recovered, points +16.6 [+9.3, +23.8]; tol. mIoU -1.4 [-2.2, -0.5]; valid, points -5.3 [-9.3, -1.3].

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 302 | 76% | 2% | **14%** [10, 18] | **14%** [10, 18] | 0.17 | 0.75 | 39.3 -> 40.6 | 5% -> 7% | 24.7 (8.4) | 0.79 | 3.3 |
| locator + B4 | 302 | 91% | 75% | **34%** [29, 40] | **43%** [38, 49] | 0.48 | 0.68 | 39.3 -> 41.6 | 5% -> 16% | 17.0 (8.4) | 1.92 | 3.8 |
| hybrid v1 auto | 302 | 91% | 75% | **35%** [30, 41] | **46%** [41, 52] | 0.49 | 0.72 | 39.3 -> 43.2 | 5% -> 15% | 18.2 (8.4) | 1.99 | 6.4 |
| hybrid v2 auto | 302 | 91% | 75% | **37%** [32, 42] | **51%** [45, 57] | 0.52 | 0.82 | 39.3 -> 46.3 | 5% -> 11% | 18.9 (8.4) | 2.28 | 6.0 |
| hybrid v3 auto | 302 | 93% | 85% | **51%** [45, 57] | **73%** [68, 78] | 0.77 | 0.77 | 39.3 -> 45.0 | 5% -> 12% | 18.2 (8.4) | 2.30 | 6.8 |
| shipped manual | 302 | 100% | 0% | **41%** [36, 47] | **46%** [41, 52] | 0.54 | 0.66 | 39.3 -> 42.7 | 5% -> 11% | 18.2 (8.4) | 0.49 | 1.4 |
| hybrid v1 manual | 302 | 100% | 0% | **43%** [38, 49] | **59%** [53, 64] | 0.62 | 0.74 | 39.3 -> 45.6 | 5% -> 8% | 19.8 (8.4) | 0.49 | 4.6 |
| hybrid v2 manual | 302 | 100% | 0% | **45%** [39, 50] | **66%** [60, 71] | 0.65 | 0.84 | 39.3 -> 48.4 | 5% -> 9% | 20.1 (8.4) | 0.45 | 4.5 |
| hybrid v3 manual | 302 | 100% | 0% | **62%** [57, 68] | **81%** [77, 85] | 0.88 | 0.76 | 39.3 -> 46.7 | 5% -> 4% | 23.5 (8.4) | 0.83 | 4.5 |

hybrid v2 auto minus shipped auto (paired, 302 cases): fixed, points +23.2 [+18.2, +28.5]; recovered, points +37.1 [+31.5, +42.7]; tol. mIoU +5.7 [+4.6, +6.9]; valid, points +4.3 [+1.0, +7.6].

hybrid v1 auto minus shipped auto (paired, 302 cases): fixed, points +21.9 [+16.9, +26.8]; recovered, points +32.5 [+27.2, +38.1]; tol. mIoU +2.6 [+1.2, +4.0]; valid, points +7.9 [+4.0, +12.3].

hybrid v2 auto minus locator + B4 (paired, 302 cases): fixed, points +2.6 [-2.0, +7.3]; recovered, points +7.9 [+3.3, +12.6]; tol. mIoU +4.7 [+3.6, +5.8]; valid, points -5.0 [-8.6, -1.3].

hybrid v2 auto minus hybrid v1 auto (paired, 302 cases): fixed, points +1.3 [-3.0, +5.6]; recovered, points +4.6 [+0.3, +9.3]; tol. mIoU +3.2 [+2.2, +4.2]; valid, points -3.6 [-7.3, -0.3].

hybrid v2 manual minus shipped manual (paired, 302 cases): fixed, points +3.6 [-2.0, +8.9]; recovered, points +19.5 [+13.6, +25.2]; tol. mIoU +5.7 [+4.5, +6.9]; valid, points -2.6 [-5.6, +0.3].

hybrid v2 manual minus hybrid v1 manual (paired, 302 cases): fixed, points +1.7 [-3.0, +6.3]; recovered, points +7.0 [+2.0, +11.9]; tol. mIoU +2.7 [+1.8, +3.7]; valid, points +0.7 [-1.7, +3.0].

hybrid v3 auto minus shipped auto (paired, 302 cases): fixed, points +37.4 [+32.1, +43.0]; recovered, points +59.3 [+53.6, +64.6]; tol. mIoU +4.4 [+3.1, +5.6]; valid, points +5.0 [+1.7, +8.3].

hybrid v3 auto minus hybrid v2 auto (paired, 302 cases): fixed, points +14.2 [+9.3, +19.2]; recovered, points +22.2 [+16.6, +27.8]; tol. mIoU -1.4 [-2.4, -0.3]; valid, points +0.7 [-3.0, +4.3].

hybrid v3 manual minus shipped manual (paired, 302 cases): fixed, points +21.2 [+15.6, +26.8]; recovered, points +35.1 [+29.1, +41.1]; tol. mIoU +4.0 [+2.9, +5.1]; valid, points -7.0 [-10.3, -4.0].

hybrid v3 manual minus hybrid v2 manual (paired, 302 cases): fixed, points +17.5 [+12.6, +22.5]; recovered, points +15.6 [+10.6, +20.5]; tol. mIoU -1.7 [-2.4, -1.0]; valid, points -4.3 [-7.3, -1.7].

## By omission kind (fixed)

| kind | n | shipped auto | locator + B4 | hybrid v1 auto | hybrid v2 auto | hybrid v3 auto | shipped manual | hybrid v1 manual | hybrid v2 manual | hybrid v3 manual |
|---|---|---|---|---|---|---|---|---|---|---|
| wall | 24 | 54% | - | 58% | 54% | 54% | - | - | - | - |
| room (whole) | 115 | 11% | 27% | 30% | 35% | 52% | 26% | 33% | 38% | 68% |
| region | 187 | 15% | 39% | 39% | 38% | 50% | 50% | 49% | 49% | 59% |
| window | 26 | 65% | - | 65% | 65% | 65% | - | - | - | - |
| door (a) | 18 | 28% | - | 28% | 28% | 28% | - | - | - | - |
| room | 22 | 73% | - | 73% | 73% | 68% | - | - | - | - |
| railing | 19 | 0% | - | 0% | 0% | 0% | - | - | - | - |
| door (b) | 23 | 48% | - | 48% | 48% | 48% | - | - | - | - |
| fixture | 19 | 5% | - | 5% | 5% | 11% | - | - | - | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| shipped auto | 151 | 0.64 | 0.01 | 1% | 0.68 |
| hybrid v1 auto | 151 | 0.74 | 0.15 | 13% | 1.38 |
| hybrid v2 auto | 151 | 0.74 | 0.15 | 13% | 1.69 |
| hybrid v3 auto | 151 | 0.70 | 0.08 | 7% | 1.02 |

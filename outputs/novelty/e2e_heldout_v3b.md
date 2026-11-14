# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_v3b

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 62% | 5% | **42%** [34, 50] | **55%** [35, 73] | 0.69 | 0.75 | 63.0 -> 65.4 | 12% -> 11% | 9.3 (3.4) | 1.16 | 3.6 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 91% | 79% | **54%** [46, 62] | **72%** [64, 78] | 0.75 | 0.79 | 41.1 -> 45.9 | 4% -> 11% | 14.4 (6.5) | 1.76 | 6.5 |
| hybrid manual | 151 | 100% | 0% | **69%** [61, 76] | **84%** [77, 89] | 0.89 | 0.79 | 41.1 -> 48.2 | 4% -> 3% | 18.8 (6.5) | 0.44 | 4.5 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 96% | 91% | **48%** [40, 56] | **75%** [67, 81] | 0.79 | 0.74 | 37.4 -> 44.0 | 5% -> 13% | 22.0 (10.2) | 2.84 | 7.2 |
| hybrid manual | 151 | 100% | 0% | **56%** [48, 63] | **79%** [72, 85] | 0.86 | 0.73 | 37.4 -> 45.2 | 5% -> 5% | 28.1 (10.2) | 1.21 | 4.5 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 302 | 93% | 85% | **51%** [45, 57] | **73%** [68, 78] | 0.77 | 0.77 | 39.3 -> 45.0 | 5% -> 12% | 18.2 (8.4) | 2.30 | 6.8 |
| hybrid manual | 302 | 100% | 0% | **62%** [57, 68] | **81%** [77, 85] | 0.88 | 0.76 | 39.3 -> 46.7 | 5% -> 4% | 23.5 (8.4) | 0.83 | 4.5 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 24 | 54% | - |
| room (whole) | 115 | 52% | 68% |
| region | 187 | 50% | 59% |
| window | 26 | 65% | - |
| door (a) | 18 | 28% | - |
| room | 22 | 68% | - |
| railing | 19 | 0% | - |
| door (b) | 23 | 48% | - |
| fixture | 19 | 11% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 151 | 0.70 | 0.08 | 7% | 1.02 |

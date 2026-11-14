# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_v3c

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 62% | 7% | **41%** [34, 49] | **55%** [35, 73] | 0.70 | 0.77 | 63.0 -> 65.0 | 12% -> 12% | 9.4 (3.4) | 1.09 | 3.4 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 95% | 87% | **57%** [49, 65] | **79%** [72, 85] | 0.84 | 0.80 | 41.1 -> 46.9 | 4% -> 13% | 14.5 (6.5) | 2.07 | 6.7 |
| hybrid manual | 151 | 100% | 0% | **69%** [61, 76] | **83%** [77, 89] | 0.91 | 0.79 | 41.1 -> 48.5 | 4% -> 4% | 18.5 (6.5) | 0.38 | 4.4 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 99% | 94% | **55%** [47, 63] | **80%** [73, 86] | 0.83 | 0.75 | 37.4 -> 44.3 | 5% -> 10% | 22.3 (10.2) | 2.99 | 7.2 |
| hybrid manual | 151 | 100% | 0% | **60%** [52, 68] | **83%** [77, 89] | 0.89 | 0.74 | 37.4 -> 45.5 | 5% -> 5% | 28.1 (10.2) | 1.26 | 4.5 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 302 | 97% | 91% | **56%** [50, 61] | **80%** [75, 84] | 0.84 | 0.77 | 39.3 -> 45.6 | 5% -> 11% | 18.4 (8.4) | 2.53 | 6.9 |
| hybrid manual | 302 | 100% | 0% | **65%** [59, 70] | **83%** [79, 87] | 0.90 | 0.77 | 39.3 -> 47.0 | 5% -> 5% | 23.3 (8.4) | 0.82 | 4.5 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 24 | 50% | - |
| room (whole) | 115 | 58% | 70% |
| region | 187 | 55% | 61% |
| window | 26 | 65% | - |
| door (a) | 18 | 28% | - |
| room | 22 | 73% | - |
| railing | 19 | 0% | - |
| door (b) | 23 | 48% | - |
| fixture | 19 | 5% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 151 | 0.66 | 0.03 | 3% | 0.93 |

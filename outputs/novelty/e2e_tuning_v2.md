# End to end on drawings: shipped pipeline vs hybrid pipeline_tuning_v2

100 val drawings (TUNING drawings: used to choose settings, so optimistic). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 68% | 10% | **50%** [40, 60] | **53%** [30, 75] | 0.74 | 0.75 | 60.1 -> 63.2 | 9% -> 10% | 10.4 (3.8) | 1.75 | 5.0 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 83% | 67% | **34%** [25, 44] | **47%** [38, 57] | 0.50 | 0.87 | 37.1 -> 45.4 | 3% -> 13% | 17.1 (7.1) | 2.63 | 6.9 |
| hybrid manual | 100 | 100% | 0% | **51%** [41, 61] | **68%** [58, 76] | 0.69 | 0.86 | 37.1 -> 47.7 | 3% -> 6% | 19.2 (7.1) | 0.45 | 4.7 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 99% | 85% | **30%** [22, 40] | **47%** [38, 57] | 0.49 | 0.76 | 36.4 -> 45.6 | 6% -> 25% | 22.5 (12.0) | 2.25 | 7.5 |
| hybrid manual | 100 | 100% | 0% | **33%** [25, 43] | **56%** [46, 65] | 0.58 | 0.75 | 36.4 -> 45.5 | 6% -> 11% | 23.8 (12.0) | 0.68 | 4.7 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 200 | 91% | 76% | **32%** [26, 39] | **47%** [40, 54] | 0.49 | 0.81 | 36.7 -> 45.5 | 4% -> 19% | 19.8 (9.6) | 2.44 | 7.2 |
| hybrid manual | 200 | 100% | 0% | **42%** [35, 49] | **62%** [55, 68] | 0.64 | 0.81 | 36.7 -> 46.6 | 4% -> 8% | 21.5 (9.6) | 0.56 | 4.7 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 17 | 41% | - |
| room (whole) | 82 | 29% | 33% |
| region | 118 | 34% | 48% |
| window | 20 | 75% | - |
| door (a) | 11 | 45% | - |
| room | 15 | 80% | - |
| fixture | 12 | 25% | - |
| door (b) | 14 | 57% | - |
| railing | 11 | 0% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 100 | 1.03 | 0.20 | 16% | 1.63 |

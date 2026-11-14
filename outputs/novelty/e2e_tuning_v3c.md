# End to end on drawings: shipped pipeline vs hybrid pipeline_tuning_v3c

100 val drawings (TUNING drawings: used to choose settings, so optimistic). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 67% | 7% | **50%** [40, 60] | **53%** [30, 75] | 0.66 | 0.77 | 60.1 -> 63.0 | 9% -> 12% | 10.2 (3.8) | 1.08 | 4.2 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 89% | 79% | **49%** [39, 59] | **66%** [56, 75] | 0.73 | 0.78 | 37.1 -> 43.4 | 3% -> 8% | 16.7 (7.1) | 3.07 | 7.5 |
| hybrid manual | 100 | 100% | 0% | **68%** [58, 76] | **83%** [74, 89] | 0.89 | 0.78 | 37.1 -> 44.9 | 3% -> 6% | 22.3 (7.1) | 0.81 | 4.7 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 98% | 93% | **47%** [38, 57] | **74%** [65, 82] | 0.79 | 0.75 | 36.4 -> 41.8 | 6% -> 11% | 24.8 (12.0) | 2.15 | 7.8 |
| hybrid manual | 100 | 100% | 0% | **51%** [41, 61] | **76%** [67, 83] | 0.84 | 0.70 | 36.4 -> 42.9 | 6% -> 8% | 28.6 (12.0) | 1.13 | 4.7 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 200 | 94% | 86% | **48%** [41, 55] | **70%** [63, 76] | 0.76 | 0.76 | 36.7 -> 42.6 | 4% -> 10% | 20.7 (9.6) | 2.61 | 7.7 |
| hybrid manual | 200 | 100% | 0% | **60%** [53, 66] | **80%** [73, 85] | 0.86 | 0.74 | 36.7 -> 43.9 | 4% -> 7% | 25.5 (9.6) | 0.97 | 4.7 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 17 | 47% | - |
| room (whole) | 82 | 52% | 63% |
| region | 118 | 45% | 57% |
| window | 20 | 75% | - |
| door (a) | 11 | 45% | - |
| room | 15 | 80% | - |
| fixture | 12 | 25% | - |
| door (b) | 14 | 50% | - |
| railing | 11 | 0% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 100 | 0.89 | 0.03 | 3% | 1.05 |

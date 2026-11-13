# End to end on drawings: shipped pipeline vs hybrid pipeline_tuning_v3b

100 val drawings (TUNING drawings: used to choose settings, so optimistic). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 67% | 4% | **51%** [41, 61] | **53%** [30, 75] | 0.76 | 0.74 | 60.1 -> 63.1 | 9% -> 11% | 10.2 (3.8) | 1.25 | 3.8 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 87% | 75% | **43%** [34, 53] | **64%** [54, 73] | 0.68 | 0.78 | 37.1 -> 43.4 | 3% -> 10% | 17.0 (7.1) | 2.98 | 7.3 |
| hybrid manual | 100 | 100% | 0% | **67%** [57, 75] | **83%** [74, 89] | 0.88 | 0.78 | 37.1 -> 45.0 | 3% -> 7% | 22.5 (7.1) | 0.86 | 4.8 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 98% | 91% | **46%** [37, 56] | **70%** [60, 78] | 0.76 | 0.75 | 36.4 -> 42.0 | 6% -> 14% | 24.2 (12.0) | 2.39 | 7.6 |
| hybrid manual | 100 | 100% | 0% | **52%** [42, 62] | **74%** [65, 82] | 0.82 | 0.70 | 36.4 -> 43.2 | 6% -> 7% | 28.9 (12.0) | 1.32 | 4.8 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 200 | 92% | 83% | **44%** [38, 51] | **67%** [60, 73] | 0.72 | 0.76 | 36.7 -> 42.7 | 4% -> 12% | 20.6 (9.6) | 2.69 | 7.5 |
| hybrid manual | 200 | 100% | 0% | **60%** [53, 66] | **78%** [72, 84] | 0.85 | 0.74 | 36.7 -> 44.1 | 4% -> 7% | 25.7 (9.6) | 1.09 | 4.8 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 17 | 47% | - |
| room (whole) | 82 | 48% | 57% |
| region | 118 | 42% | 61% |
| window | 20 | 75% | - |
| door (a) | 11 | 45% | - |
| room | 15 | 80% | - |
| fixture | 12 | 25% | - |
| door (b) | 14 | 57% | - |
| railing | 11 | 0% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 100 | 0.91 | 0.06 | 6% | 1.27 |

# End to end on drawings: shipped pipeline vs hybrid pipeline_tuning_v3a

100 val drawings (TUNING drawings: used to choose settings, so optimistic). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 68% | 10% | **48%** [38, 58] | **53%** [30, 75] | 0.77 | 0.73 | 60.1 -> 63.1 | 9% -> 11% | 10.4 (3.8) | 2.39 | 4.8 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 87% | 76% | **46%** [37, 56] | **64%** [54, 73] | 0.69 | 0.77 | 37.1 -> 43.7 | 3% -> 11% | 16.3 (7.1) | 4.44 | 7.2 |
| hybrid manual | 100 | 100% | 0% | **65%** [55, 74] | **81%** [72, 87] | 0.87 | 0.78 | 37.1 -> 45.2 | 3% -> 7% | 22.3 (7.1) | 0.87 | 4.6 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 100 | 99% | 92% | **48%** [38, 58] | **72%** [63, 80] | 0.77 | 0.72 | 36.4 -> 43.2 | 6% -> 13% | 23.5 (12.0) | 4.15 | 7.5 |
| hybrid manual | 100 | 100% | 0% | **48%** [38, 58] | **72%** [63, 80] | 0.81 | 0.70 | 36.4 -> 43.4 | 6% -> 8% | 28.9 (12.0) | 1.30 | 4.6 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 200 | 93% | 84% | **47%** [40, 54] | **68%** [61, 74] | 0.73 | 0.75 | 36.7 -> 43.5 | 4% -> 12% | 19.9 (9.6) | 4.29 | 7.3 |
| hybrid manual | 200 | 100% | 0% | **56%** [50, 63] | **76%** [70, 82] | 0.84 | 0.74 | 36.7 -> 44.3 | 4% -> 8% | 25.6 (9.6) | 1.08 | 4.6 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 17 | 35% | - |
| room (whole) | 82 | 50% | 55% |
| region | 118 | 45% | 58% |
| window | 20 | 75% | - |
| door (a) | 11 | 45% | - |
| room | 15 | 73% | - |
| fixture | 12 | 25% | - |
| door (b) | 14 | 57% | - |
| railing | 11 | 0% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 100 | 1.04 | 0.21 | 18% | 2.01 |

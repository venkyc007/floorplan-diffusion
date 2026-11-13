# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_v2

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 64% | 8% | **42%** [34, 50] | **55%** [35, 73] | 0.64 | 0.80 | 63.0 -> 65.5 | 12% -> 12% | 9.2 (3.4) | 1.72 | 3.7 |

## medium omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 87% | 70% | **38%** [30, 46] | **50%** [42, 58] | 0.52 | 0.83 | 41.1 -> 47.3 | 4% -> 9% | 15.3 (6.5) | 2.33 | 5.9 |
| hybrid manual | 151 | 100% | 0% | **48%** [40, 56] | **70%** [62, 76] | 0.69 | 0.85 | 41.1 -> 50.2 | 4% -> 7% | 16.6 (6.5) | 0.25 | 4.5 |

## large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 151 | 95% | 80% | **36%** [29, 44] | **52%** [44, 59] | 0.51 | 0.80 | 37.4 -> 45.3 | 5% -> 13% | 22.5 (10.2) | 2.23 | 6.2 |
| hybrid manual | 151 | 100% | 0% | **42%** [34, 50] | **62%** [54, 70] | 0.60 | 0.82 | 37.4 -> 46.6 | 5% -> 11% | 23.5 (10.2) | 0.64 | 4.5 |

## medium + large omissions

| arm | n | found | region flag | fixed | recovered (2 px) | missing-wall recall | added-wall precision | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid auto | 302 | 91% | 75% | **37%** [32, 42] | **51%** [45, 57] | 0.52 | 0.82 | 39.3 -> 46.3 | 5% -> 11% | 18.9 (8.4) | 2.28 | 6.0 |
| hybrid manual | 302 | 100% | 0% | **45%** [39, 50] | **66%** [60, 71] | 0.65 | 0.84 | 39.3 -> 48.4 | 5% -> 9% | 20.1 (8.4) | 0.45 | 4.5 |

## By omission kind (fixed)

| kind | n | hybrid auto | hybrid manual |
|---|---|---|---|
| wall | 24 | 54% | - |
| room (whole) | 115 | 35% | 38% |
| region | 187 | 38% | 49% |
| window | 26 | 65% | - |
| door (a) | 18 | 28% | - |
| room | 22 | 73% | - |
| railing | 19 | 0% | - |
| door (b) | 23 | 48% | - |
| fixture | 19 | 5% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| hybrid auto | 151 | 0.74 | 0.15 | 13% | 1.69 |

# End to end on drawings: shipped pipeline vs hybrid pipeline_heldout_base

151 val drawings (held-out: never used to choose a setting). One omission per case, erased on the drawing; the automatic arms are given nothing. fixed = found and wall IoU >= 0.3 at the erased structure (src/stage2/pipeline/e2e_val.score, as in the Stage 2 test). Brackets: Wilson 95% intervals. Script: src/novelty/eval_e2e.py.

## small omissions

| arm | n | found | region flag | fixed | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 61% | 1% | **42%** [34, 50] | 63.0 -> 65.6 | 12% -> 13% | 9.2 (3.4) | 0.74 | 3.0 |

## medium omissions

| arm | n | found | region flag | fixed | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 73% | 1% | **15%** [10, 21] | 41.1 -> 42.6 | 4% -> 7% | 19.9 (6.5) | 0.69 | 3.2 |
| shipped manual | 151 | 100% | 0% | **47%** [39, 55] | 41.1 -> 46.0 | 4% -> 9% | 15.3 (6.5) | 0.22 | 1.3 |

## large omissions

| arm | n | found | region flag | fixed | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 151 | 80% | 3% | **13%** [8, 19] | 37.4 -> 38.6 | 5% -> 7% | 29.6 (10.2) | 0.89 | 3.1 |
| shipped manual | 151 | 100% | 0% | **35%** [28, 43] | 37.4 -> 39.4 | 5% -> 13% | 21.1 (10.2) | 0.75 | 1.5 |

## medium + large omissions

| arm | n | found | region flag | fixed | tol. mIoU before -> after | valid before -> after | violations (truth) | false additions / drawing | s / drawing |
|---|---|---|---|---|---|---|---|---|---|
| shipped auto | 302 | 76% | 2% | **14%** [10, 18] | 39.3 -> 40.6 | 5% -> 7% | 24.7 (8.4) | 0.79 | 3.1 |
| shipped manual | 302 | 100% | 0% | **41%** [36, 47] | 39.3 -> 42.7 | 5% -> 11% | 18.2 (8.4) | 0.49 | 1.4 |

## By omission kind (fixed)

| kind | n | shipped auto | shipped manual |
|---|---|---|---|
| wall | 24 | 54% | - |
| room (whole) | 115 | 11% | 26% |
| region | 187 | 15% | 50% |
| window | 26 | 65% | - |
| door (a) | 18 | 28% | - |
| room | 22 | 73% | - |
| railing | 19 | 0% | - |
| door (b) | 23 | 48% | - |
| fixture | 19 | 5% | - |

## Intact drawings (false alarms)

| arm | n | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|---|
| shipped auto | 151 | 0.64 | 0.01 | 1% | 0.68 |

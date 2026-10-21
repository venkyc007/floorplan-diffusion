# B4 validation with the hole given (val windows; current models vs B4)

30 large-hole windows (256 px) + 30 small-hole windows (128 px) from completion_val, seed 77; fast sampling, eta 0. Scores inside the hole (see src/stage2/complete/b4_val.py).

| holes | arm | fixed (IoU >= 0.3) | tol. mIoU | valid | s per hole |
|---|---|---|---|---|---|
| large | current, B1 big window | 26 / 30 | 41.9 | 30% | 33.6 |
| large | current, B3 tiled | 22 / 30 | 38.1 | 37% | 32.0 |
| large | B4, B1 big window | 29 / 30 | 49.3 | 47% | 33.7 |
| small | current | 25 / 30 | 69.2 | 63% | 18.9 |
| small | B4 | 25 / 30 | 68.5 | 67% | 18.9 |

**Small-hole routing:** current 69.2 / 63% valid vs B4 68.5 / 67% -> B4 is worse on small holes on at least one of tolerant mIoU / validity: small holes stay on the current models.

By hole kind (fixed / tol. mIoU per arm):

| holes | kind | n | arms |
|---|---|---|---|
| large | region_large | 18 | current, B1 big window: 16 / 41.9; current, B3 tiled: 11 / 37.2; B4, B1 big window: 17 / 50.6 |
| large | room_whole | 12 | current, B1 big window: 10 / 41.9; current, B3 tiled: 11 / 39.6; B4, B1 big window: 12 / 47.3 |
| small | door | 2 | current: 1 / 76.0; B4: 2 / 75.2 |
| small | fixture | 2 | current: 1 / 55.3; B4: 1 / 50.4 |
| small | railing | 7 | current: 5 / 71.7; B4: 4 / 71.1 |
| small | region | 3 | current: 3 / 69.4; B4: 3 / 83.6 |
| small | room | 8 | current: 8 / 47.5; B4: 8 / 45.2 |
| small | wall | 2 | current: 2 / 90.7; B4: 2 / 90.7 |
| small | window | 6 | current: 5 / 90.4; B4: 5 / 85.5 |

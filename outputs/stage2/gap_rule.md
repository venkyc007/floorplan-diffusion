# Doorless-gap rule R5: final cut-offs (tuning plans only)

Tuning plans: 212 (100 intact, 18 door kind (b) omissions). Room-size cut-offs: need_area = p25 of 371 bedroom areas = 1514 px; small_other = p25 of 769 other_room areas = 460 px (completion scale, walls 3 px).
Pair rule alone (rate < 10%, no size condition): 0.23 false alarms per intact plan.
Budget for R5: 0.4 - 0.31 (detector + C8, held-out) = 0.09 per intact plan.

| shortest gap (px) | R5 false alarms / intact plan | door (b) recall |
|---|---|---|
| 5 | 0.13 | 44% |
| 6 | 0.12 | 44% |
| 7 | 0.12 | 44% |
| 8 | 0.12 | 44% |
| 9 | 0.12 | 44% |
| 10 | 0.10 | 39% |
| 11 | 0.09 | 33% |

Chosen: shortest gap 11 px -> 0.09 false alarms, door (b) recall 33% (tuning plans).

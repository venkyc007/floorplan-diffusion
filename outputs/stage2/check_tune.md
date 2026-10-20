# CHECK tuned on parsed TUNING drawings (parser v2B, detectors v2 + v3, erasure v2; + window / wall ink double checks, 2026-10-20)

40 intact + 160 omissions on tuning drawings. Budget 0.4 false alarms per intact drawing in total; rules + ink chosen first within 0.25, then the detector inside what is left (user, 2026-10-19). Found = a proposal overlaps the omission. Ink column: tier-1 threshold (R5's door pairs) / tier-2 threshold (other interior doors).

## Calibration of the window / wall double checks (tuning drawings)

Percentiles 10 / 50 / 90. Intact = every parsed window (wall run) on the intact drawings; omission = the parsed windows overlapping a window omission (the parser kept a window there), the longest uninked wall run overlapping a wall omission (0 = none: the parser left the gap or the wall is inked).

| feature | intact components | at omissions | n intact / n at omissions |
|---|---|---|---|
| window d_wall | 0.103 / 0.242 / 0.454 | 0.024 / 0.084 / 0.239 | 206 / 16 |
| window ink | 0.250 / 0.505 / 0.683 | 0.395 / 0.651 / 0.801 | 206 / 16 |
| window runs | 1.000 / 2.000 / 3.000 | 1.000 / 1.500 / 3.000 | 206 / 16 |
| uninked wall run length (px, runs >= 4) | 19.800 / 51.000 / 82.200 | 0.000 / 0.000 / 0.000 | 2 / 23 |
| share of parsed wall px uninked, per drawing | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.005 | |

Window omissions with a parsed window at the omission: 16 of 23; uninked wall runs >= 4 px at a wall omission: 1 of 23.

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 57% | 43% | 74% | 74% | 61% | 0% | 55% | 95% |

## Which detector (best within the budget with the chosen rules)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (v2) v2 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.35 | 31% | 26% | 43% | 43% | 30% | 0% | 16% | 59% |
| (v3) v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 53% | 43% | 70% | 57% | 57% | 0% | 52% | 95% |
| (both) v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 57% | 43% | 74% | 74% | 61% | 0% | 55% | 95% |

## What the window / wall ink checks, R6 and R7 add (the choice vs the choice with that part off)

| part | recall gained (points) | false alarms cost | the choice without it |
|---|---|---|---|
| window ink check (parse vs drawing) | +1.2 | +0.05 | 0.25 false alarms, 56% recall |
| wall ink check (parse vs drawing) | +0.0 | +0.05 | 0.25 false alarms, 57% recall |
| R6 (windowless exterior span) | +0.6 | +0.02 | 0.28 false alarms, 57% recall |
| R7 (gap too narrow) | +0.0 | +0.00 | 0.30 false alarms, 57% recall |

## R6 / R7 / window ink / wall ink on their own

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (R6 alone) off | off | off | off | x1.0 | off | off | off | 0.17 | 1% | 0% | 0% | 4% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.2 | off | off | off | 0.03 | 1% | 0% | 0% | 4% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.5 | off | off | off | 0.00 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (R7 alone) off | off | off | off | off | >= 1 px | off | off | 0.17 | 3% | 0% | 0% | 4% | 0% | 7% | 0% | 9% |
| (R7 alone) off | off | off | off | off | >= 2 px | off | off | 0.17 | 2% | 0% | 0% | 4% | 0% | 7% | 0% | 5% |
| (R7 alone) off | off | off | off | off | >= 3 px | off | off | 0.07 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0 | off | 0.07 | 1% | 0% | 0% | 9% | 0% | 0% | 0% | 0% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0.02 | off | 0.07 | 4% | 0% | 0% | 9% | 17% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0.05 | off | 0.07 | 4% | 0% | 0% | 9% | 17% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0 | off | 0.03 | 2% | 4% | 0% | 13% | 0% | 0% | 0% | 0% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0.02 | off | 0.03 | 6% | 4% | 0% | 13% | 17% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0.05 | off | 0.03 | 6% | 4% | 0% | 13% | 17% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0 | off | 0.05 | 4% | 4% | 0% | 22% | 0% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0.02 | off | 0.05 | 8% | 4% | 0% | 22% | 17% | 0% | 0% | 9% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0.05 | off | 0.05 | 8% | 4% | 0% | 22% | 17% | 0% | 0% | 9% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0 | off | 0.12 | 4% | 4% | 0% | 22% | 0% | 0% | 0% | 5% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0.02 | off | 0.12 | 8% | 4% | 0% | 22% | 17% | 0% | 0% | 9% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0.05 | off | 0.12 | 8% | 4% | 0% | 22% | 17% | 0% | 0% | 9% |
| (wall ink alone) off | off | off | off | off | off | off | >= 4 px | 0.05 | 3% | 4% | 0% | 0% | 4% | 0% | 0% | 14% |
| (wall ink alone) off | off | off | off | off | off | off | >= 6 px | 0.05 | 3% | 0% | 0% | 0% | 4% | 0% | 0% | 14% |
| (wall ink alone) off | off | off | off | off | off | off | >= 9 px | 0.05 | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 5% |
| (wall ink alone) off | off | off | off | off | off | off | >= 13 px | 0.03 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Stage 1 (rules + ink, detector off; budget 0.25), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.25 | 21% | 26% | 35% | 26% | 26% | 0% | 6% | 27% |
| off | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.05 | >= 4 px | 0.25 | 21% | 26% | 35% | 26% | 26% | 0% | 6% | 27% |
| off | 700 | on | 2.0 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | off | 0.25 | 20% | 26% | 35% | 26% | 22% | 0% | 6% | 27% |
| off | 700 | on | 2.0 / off | x1.2 | off | d_wall < 0.08 / ink < 0.05 | off | 0.25 | 20% | 26% | 35% | 26% | 22% | 0% | 6% | 27% |
| off | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.23 | 20% | 26% | 35% | 22% | 26% | 0% | 6% | 27% |
| off | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.05 | >= 4 px | 0.23 | 20% | 26% | 35% | 22% | 26% | 0% | 6% | 27% |
| off | 700 | on | 1.5 / off | off | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.23 | 20% | 26% | 35% | 22% | 26% | 0% | 6% | 27% |
| off | 700 | on | 1.5 / off | off | off | d_wall < 0.08 / ink < 0.05 | >= 4 px | 0.23 | 20% | 26% | 35% | 22% | 26% | 0% | 6% | 27% |

## Stage 2 (detector added to the chosen rules; budget 0.4), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 57% | 43% | 74% | 74% | 61% | 0% | 55% | 95% |
| v2 0.6 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.33 | 57% | 43% | 74% | 74% | 61% | 0% | 55% | 95% |
| v2 0.5 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.38 | 57% | 43% | 74% | 74% | 61% | 0% | 55% | 95% |
| v2 0.6 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 57% | 43% | 70% | 74% | 61% | 0% | 55% | 95% |
| v2 0.8 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 57% | 43% | 74% | 74% | 57% | 0% | 55% | 95% |
| v2 0.7 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.33 | 57% | 43% | 74% | 74% | 57% | 0% | 55% | 95% |
| v2 0.5 / 40 px + v3 0.6 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 56% | 39% | 70% | 74% | 61% | 0% | 55% | 95% |
| v2 0.9 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | >= 4 px | 0.30 | 56% | 43% | 70% | 74% | 57% | 0% | 55% | 95% |

## Detector alone (best within 0.15 / 0.25 / 0.35 false alarms)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 20 px | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |
| v2 0.5 / 20 px | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |
| v2 0.5 / 20 px | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |

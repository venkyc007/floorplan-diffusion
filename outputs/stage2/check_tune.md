# CHECK tuned on parsed TUNING drawings (parser v2B, detectors v2 + v3, erasure v2; + window / wall ink double checks, 2026-10-20)

40 intact + 160 omissions on tuning drawings. Budget 0.4 false alarms per intact drawing in total; rules + ink chosen first within 0.25, then the detector inside what is left (user, 2026-10-19). Found = a proposal overlaps the omission. Ink column: tier-1 threshold (R5's door pairs) / tier-2 threshold (other interior doors).

## Calibration of the window / wall double checks (tuning drawings)

Percentiles 10 / 50 / 90. Intact = every parsed window (wall run) on the intact drawings; omission = the parsed windows overlapping a window omission (the parser kept a window there), the longest uninked wall run overlapping a wall omission (0 = none: the parser left the gap or the wall is inked).

| feature | intact components | at omissions | n intact / n at omissions |
|---|---|---|---|
| window d_wall | 0.103 / 0.242 / 0.454 | 0.012 / 0.078 / 0.132 | 206 / 14 |
| window ink | 0.250 / 0.505 / 0.683 | 0.559 / 0.588 / 0.753 | 206 / 14 |
| window runs | 1.000 / 2.000 / 3.000 | 1.000 / 1.000 / 2.000 | 206 / 14 |
| uninked wall run length (px, runs >= 4) | 19.800 / 51.000 / 82.200 | 0.000 / 0.000 / 0.000 | 2 / 18 |
| share of parsed wall px uninked, per drawing | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.008 | |

Window omissions with a parsed window at the omission: 14 of 18; uninked wall runs >= 4 px at a wall omission: 0 of 18.

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.6 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.30 | 71% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 76% |

## Which detector (best within the budget with the chosen rules)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (v2) v2 0.6 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 33% | 28% | 22% | 39% | 28% | 0% | 17% | 53% | 56% | 53% |
| (v3) v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 68% | 72% | 72% | 78% | 83% | 0% | 54% | 100% | 89% | 65% |
| (both) v2 0.6 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.30 | 71% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 76% |

## What the window / wall ink checks, R6 and R7 add (the choice vs the choice with that part off)

| part | recall gained (points) | false alarms cost | the choice without it |
|---|---|---|---|
| R8 merged room (region missing) | +1.1 | +0.05 | 0.25 false alarms, 70% recall |
| R5 doorless gap (forced on) | +0.0 | +0.05 | 0.25 false alarms, 71% recall |
| window ink check (parse vs drawing) | +1.2 | +0.05 | 0.25 false alarms, 70% recall |
| wall ink check (parse vs drawing) | +0.0 | +0.00 | 0.30 false alarms, 71% recall |
| R6 (windowless exterior span) | +0.0 | +0.00 | 0.30 false alarms, 71% recall |
| R7 (gap too narrow) | +0.0 | +0.00 | 0.30 false alarms, 71% recall |

## R6 / R7 / window ink / wall ink on their own

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (R6 alone) off | off | off | off | x1.0 | off | off | off | off | 0.17 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.2 | off | off | off | off | 0.03 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.5 | off | off | off | off | 0.00 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (R7 alone) off | off | off | off | off | >= 1 px | off | off | off | 0.17 | 1% | 0% | 0% | 0% | 0% | 8% | 0% | 0% | 0% | 0% |
| (R7 alone) off | off | off | off | off | >= 2 px | off | off | off | 0.17 | 1% | 0% | 0% | 0% | 0% | 8% | 0% | 0% | 0% | 0% |
| (R7 alone) off | off | off | off | off | >= 3 px | off | off | off | 0.07 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0 | off | off | 0.07 | 6% | 0% | 0% | 11% | 0% | 0% | 0% | 0% | 17% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0.02 | off | off | 0.07 | 10% | 0% | 0% | 11% | 11% | 0% | 0% | 24% | 17% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.03 / ink < 0.05 | off | off | 0.07 | 11% | 0% | 0% | 11% | 22% | 0% | 0% | 24% | 17% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0 | off | off | 0.03 | 8% | 6% | 0% | 17% | 0% | 0% | 0% | 0% | 22% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0.02 | off | off | 0.03 | 11% | 6% | 0% | 11% | 11% | 0% | 0% | 24% | 22% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.05 / ink < 0.05 | off | off | 0.03 | 12% | 6% | 0% | 11% | 22% | 0% | 0% | 24% | 22% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0 | off | off | 0.05 | 11% | 6% | 0% | 33% | 0% | 0% | 0% | 12% | 28% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.05 | 14% | 6% | 0% | 28% | 11% | 0% | 0% | 29% | 28% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.08 / ink < 0.05 | off | off | 0.05 | 15% | 6% | 0% | 28% | 22% | 0% | 0% | 29% | 28% | 24% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0 | off | off | 0.12 | 10% | 6% | 0% | 39% | 0% | 0% | 0% | 0% | 17% | 29% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0.02 | off | off | 0.12 | 13% | 6% | 0% | 33% | 11% | 0% | 0% | 18% | 17% | 29% |
| (window ink alone) off | off | off | off | off | off | d_wall < 0.12 / ink < 0.05 | off | off | 0.12 | 14% | 6% | 0% | 33% | 22% | 0% | 0% | 18% | 17% | 29% |
| (wall ink alone) off | off | off | off | off | off | off | >= 4 px | off | 0.05 | 6% | 0% | 0% | 0% | 0% | 0% | 0% | 6% | 22% | 24% |
| (R8 alone) off | off | off | off | off | off | off | off | mix | 0.05 | 2% | 0% | 0% | 0% | 0% | 0% | 4% | 0% | 0% | 12% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.5 x type / 2.0 x all | 1.60 | 68% | 44% | 83% | 61% | 56% | 42% | 58% | 82% | 89% | 94% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.5 x type / 2.0 x all (not living / hallway) | 1.35 | 44% | 22% | 50% | 28% | 44% | 17% | 50% | 53% | 78% | 53% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.5 x type / 2.0 x all + mix | 1.62 | 68% | 44% | 83% | 61% | 56% | 42% | 58% | 82% | 89% | 94% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.5 x type / 2.0 x all (not living / hallway) + mix | 1.40 | 44% | 22% | 50% | 28% | 44% | 17% | 50% | 53% | 78% | 53% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.8 x type / 2.5 x all | 1.40 | 61% | 28% | 67% | 50% | 50% | 42% | 54% | 82% | 89% | 88% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.8 x type / 2.5 x all (not living / hallway) | 1.15 | 40% | 17% | 44% | 28% | 44% | 17% | 42% | 41% | 78% | 53% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.8 x type / 2.5 x all + mix | 1.43 | 61% | 28% | 67% | 50% | 50% | 42% | 54% | 82% | 89% | 88% |
| (R8 alone) off | off | off | off | off | off | off | off | > 1.8 x type / 2.5 x all (not living / hallway) + mix | 1.20 | 40% | 17% | 44% | 28% | 44% | 17% | 42% | 41% | 78% | 53% |
| (R8 alone) off | off | off | off | off | off | off | off | > 2.2 x type / 3.0 x all | 1.23 | 55% | 28% | 56% | 33% | 39% | 33% | 50% | 76% | 89% | 88% |
| (R8 alone) off | off | off | off | off | off | off | off | > 2.2 x type / 3.0 x all (not living / hallway) | 1.00 | 33% | 17% | 33% | 22% | 28% | 17% | 42% | 29% | 67% | 47% |
| (R8 alone) off | off | off | off | off | off | off | off | > 2.2 x type / 3.0 x all + mix | 1.25 | 55% | 28% | 56% | 33% | 39% | 33% | 50% | 76% | 89% | 88% |
| (R8 alone) off | off | off | off | off | off | off | off | > 2.2 x type / 3.0 x all (not living / hallway) + mix | 1.05 | 33% | 17% | 33% | 22% | 28% | 17% | 42% | 29% | 67% | 47% |
| (R8 alone) off | off | off | off | off | off | off | off | > 3.0 x type / 4.0 x all | 0.72 | 42% | 17% | 33% | 17% | 22% | 33% | 33% | 53% | 83% | 82% |
| (R8 alone) off | off | off | off | off | off | off | off | > 3.0 x type / 4.0 x all (not living / hallway) | 0.55 | 24% | 11% | 17% | 6% | 11% | 17% | 21% | 24% | 61% | 47% |
| (R8 alone) off | off | off | off | off | off | off | off | > 3.0 x type / 4.0 x all + mix | 0.78 | 42% | 17% | 33% | 17% | 22% | 33% | 33% | 53% | 83% | 82% |
| (R8 alone) off | off | off | off | off | off | off | off | > 3.0 x type / 4.0 x all (not living / hallway) + mix | 0.60 | 24% | 11% | 17% | 6% | 11% | 17% | 21% | 24% | 61% | 47% |
| (R8 alone) off | off | off | off | off | off | off | off | > 4.0 x type / 5.0 x all | 0.50 | 32% | 11% | 22% | 17% | 17% | 17% | 21% | 41% | 78% | 65% |
| (R8 alone) off | off | off | off | off | off | off | off | > 4.0 x type / 5.0 x all (not living / hallway) | 0.30 | 17% | 6% | 11% | 6% | 11% | 8% | 12% | 12% | 50% | 35% |
| (R8 alone) off | off | off | off | off | off | off | off | > 4.0 x type / 5.0 x all + mix | 0.55 | 32% | 11% | 22% | 17% | 17% | 17% | 25% | 41% | 78% | 65% |
| (R8 alone) off | off | off | off | off | off | off | off | > 4.0 x type / 5.0 x all (not living / hallway) + mix | 0.35 | 17% | 6% | 11% | 6% | 11% | 8% | 17% | 12% | 50% | 35% |

## Stage 1 (rules + ink, detector off; budget 0.25), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.23 | 25% | 28% | 22% | 28% | 28% | 0% | 8% | 35% | 39% | 41% |
| off | 700 | on | 1.5 / off | off | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.23 | 25% | 28% | 22% | 28% | 28% | 0% | 8% | 35% | 39% | 41% |
| off | 700 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.25 | 25% | 28% | 22% | 28% | 28% | 0% | 8% | 35% | 39% | 41% |
| off | 400 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.23 | 24% | 33% | 22% | 28% | 28% | 0% | 4% | 35% | 39% | 29% |
| off | 400 | on | 1.5 / off | off | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.23 | 24% | 33% | 22% | 28% | 28% | 0% | 4% | 35% | 39% | 29% |
| off | 700 | on | 2.0 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.23 | 24% | 33% | 22% | 28% | 28% | 0% | 4% | 35% | 39% | 29% |
| off | 700 | on | 2.0 / off | off | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.23 | 24% | 33% | 22% | 28% | 28% | 0% | 4% | 35% | 39% | 29% |
| off | 400 | on | 1.5 / off | x1.2 | off | d_wall < 0.08 / ink < 0.02 | off | off | 0.25 | 24% | 33% | 22% | 28% | 28% | 0% | 4% | 35% | 39% | 29% |

## Stage 2 (detector added to the chosen rules; budget 0.4), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.6 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.30 | 71% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 76% |
| v2 0.5 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.35 | 71% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 76% |
| v2 0.7 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.30 | 70% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 71% |
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 70% | 72% | 72% | 89% | 83% | 0% | 54% | 100% | 89% | 71% |
| v2 0.6 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 70% | 72% | 72% | 89% | 83% | 0% | 54% | 100% | 89% | 71% |
| v2 0.8 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 70% | 72% | 72% | 89% | 83% | 0% | 58% | 100% | 89% | 65% |
| v2 0.9 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 69% | 72% | 72% | 89% | 83% | 0% | 54% | 100% | 89% | 65% |
| v2 0.95 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / off | x1.5 | off | d_wall < 0.08 / ink < 0.02 | off | mix | 0.28 | 69% | 72% | 72% | 89% | 83% | 0% | 54% | 100% | 89% | 65% |

## Detector alone (best within 0.15 / 0.25 / 0.35 false alarms)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | window ink | wall ink | R8 merged room | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room | room (whole) | region |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 20 px | off | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 0% | 17% | 0% | 0% | 8% | 24% | 28% | 24% |
| v2 0.5 / 20 px | off | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 0% | 17% | 0% | 0% | 8% | 24% | 28% | 24% |
| v2 0.5 / 20 px | off | off | off | off | off | off | off | off | 0.12 | 11% | 0% | 0% | 17% | 0% | 0% | 8% | 24% | 28% | 24% |

# CHECK tuned on parsed TUNING drawings (parser v2B, detector v2, erasure v2; + R6, R7, ink tiers)

40 intact + 160 omissions on tuning drawings. Budget 0.6 false alarms per intact drawing in total; rules + ink chosen first within 0.35, then the detector inside what is left (user, 2026-10-19). Found = a proposal overlaps the omission. Ink column: tier-1 threshold (R5's door pairs) / tier-2 threshold (other interior doors).

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.55 | 20% | 22% | 39% | 22% | 9% | 0% | 16% | 32% |

## What R6, R7 and the second ink tier add (the choice vs the choice with that part off)

| part | recall gained (points) | false alarms cost | the choice without it |
|---|---|---|---|
| R6 (windowless exterior span) | +0.6 | +0.03 | 0.53 false alarms, 19% recall |
| R7 (gap too narrow) | +0.0 | +0.00 | 0.55 false alarms, 20% recall |
| ink tier 2 (other interior doors) | +3.0 | +0.18 | 0.38 false alarms, 17% recall |

## R6 / R7 on their own

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (R6 alone) off | off | off | off | x1.0 | off | 0.17 | 1% | 0% | 0% | 4% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.2 | off | 0.03 | 1% | 0% | 0% | 4% | 0% | 0% | 0% | 0% |
| (R6 alone) off | off | off | off | x1.5 | off | 0.00 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| (R7 alone) off | off | off | off | off | >= 1 px | 0.17 | 3% | 0% | 0% | 4% | 0% | 7% | 0% | 9% |
| (R7 alone) off | off | off | off | off | >= 2 px | 0.17 | 2% | 0% | 0% | 4% | 0% | 7% | 0% | 5% |
| (R7 alone) off | off | off | off | off | >= 3 px | 0.07 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## Stage 1 (rules + ink, detector off; budget 0.35), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 15% | 22% | 39% | 4% | 9% | 0% | 10% | 23% |
| off | 700 | on | 2.0 / 0.25 | x1.2 | off | 0.35 | 15% | 26% | 39% | 4% | 9% | 0% | 6% | 18% |
| off | 700 | on | 2.0 / 0.5 | x1.2 | off | 0.35 | 15% | 26% | 39% | 4% | 9% | 0% | 6% | 18% |
| off | 700 | on | 1.5 / 1.0 | x1.5 | off | 0.33 | 15% | 22% | 39% | 0% | 9% | 0% | 10% | 23% |
| off | 700 | on | 1.5 / 1.0 | off | off | 0.33 | 15% | 22% | 39% | 0% | 9% | 0% | 10% | 23% |
| off | 200 | on | 3.0 / off | x1.5 | off | 0.35 | 15% | 30% | 35% | 0% | 4% | 0% | 10% | 23% |
| off | 200 | on | 3.0 / off | off | off | 0.35 | 15% | 30% | 35% | 0% | 4% | 0% | 10% | 23% |
| off | 200 | on | 2.0 / off | x1.2 | off | 0.33 | 15% | 30% | 35% | 4% | 4% | 0% | 10% | 18% |

## Stage 2 (detector added to the chosen rules; budget 0.6), top 8

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.55 | 20% | 22% | 39% | 22% | 9% | 0% | 16% | 32% |
| 0.95 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.47 | 19% | 22% | 39% | 17% | 9% | 0% | 16% | 32% |
| 0.8 / 40 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.60 | 18% | 22% | 39% | 9% | 9% | 0% | 16% | 32% |
| 0.5 / 80 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.60 | 18% | 22% | 39% | 4% | 9% | 0% | 13% | 36% |
| 0.98 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.45 | 17% | 22% | 39% | 13% | 9% | 0% | 16% | 23% |
| 0.8 / 80 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.45 | 17% | 22% | 39% | 4% | 9% | 0% | 13% | 32% |
| 0.9 / 40 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.47 | 17% | 22% | 39% | 4% | 9% | 0% | 13% | 32% |
| 0.7 / 80 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.55 | 17% | 22% | 39% | 4% | 9% | 0% | 13% | 32% |

## Detector alone (best within 0.15 / 0.25 / 0.35 false alarms)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.95 / 20 px | off | off | off | off | off | 0.12 | 5% | 0% | 0% | 13% | 0% | 0% | 6% | 14% |
| 0.9 / 20 px | off | off | off | off | off | 0.23 | 5% | 0% | 0% | 17% | 0% | 0% | 6% | 14% |
| 0.5 / 40 px | off | off | off | off | off | 0.30 | 9% | 0% | 4% | 17% | 4% | 0% | 13% | 27% |

# CHECK tuned on parsed TUNING drawings (parser v2B, detector v2, erasure v2; + R6, R7, ink tiers)

40 intact + 160 omissions on tuning drawings. Budget 0.6 false alarms per intact drawing in total; rules + ink chosen first within 0.35, then the detector inside what is left (user, 2026-10-19). Found = a proposal overlaps the omission. Ink column: tier-1 threshold (R5's door pairs) / tier-2 threshold (other interior doors).

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 58% | 48% | 78% | 70% | 61% | 0% | 55% | 95% |

## Which detector (best within the budget with the chosen rules)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (v2) v2 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.42 | 25% | 22% | 48% | 22% | 13% | 0% | 16% | 55% |
| (v3) v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.38 | 54% | 48% | 74% | 52% | 57% | 0% | 52% | 95% |
| (both) v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 58% | 48% | 78% | 70% | 61% | 0% | 55% | 95% |

## What R6, R7 and the second ink tier add (the choice vs the choice with that part off)

| part | recall gained (points) | false alarms cost | the choice without it |
|---|---|---|---|
| R6 (windowless exterior span) | +0.6 | +0.02 | 0.33 false alarms, 57% recall |
| R7 (gap too narrow) | +0.0 | +0.00 | 0.35 false alarms, 58% recall |
| ink tier 2 (other interior doors) | +1.9 | +0.15 | 0.20 false alarms, 56% recall |

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
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 58% | 48% | 78% | 70% | 61% | 0% | 55% | 95% |
| v2 0.6 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.38 | 58% | 48% | 78% | 70% | 61% | 0% | 55% | 95% |
| v2 0.5 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.42 | 58% | 48% | 78% | 70% | 61% | 0% | 55% | 95% |
| v2 0.6 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 57% | 48% | 74% | 70% | 61% | 0% | 55% | 95% |
| v2 0.8 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 57% | 48% | 78% | 70% | 57% | 0% | 55% | 95% |
| v2 0.7 / 20 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.38 | 57% | 48% | 78% | 70% | 57% | 0% | 55% | 95% |
| v2 0.5 / 40 px + v3 0.6 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 57% | 43% | 74% | 70% | 61% | 0% | 55% | 95% |
| v2 0.5 / 40 px + v3 0.7 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.35 | 57% | 43% | 74% | 70% | 61% | 0% | 55% | 95% |

## Detector alone (best within 0.15 / 0.25 / 0.35 false alarms)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 20 px | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |
| v2 0.5 / 20 px | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |
| v2 0.5 / 20 px | off | off | off | off | off | 0.12 | 11% | 0% | 9% | 17% | 4% | 0% | 13% | 36% |

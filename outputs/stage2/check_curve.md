# Held-out trade-off: recall vs false alarms (parser v2B, detector v2, erasure v2)

59 intact + 200 omissions (image_eval's held-out cases). Chosen setting: see 'Chosen point' at the end. Figure: outputs/stage2/check_curve.png.

## (i) rules + ink

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 2756 | off | off | off | off | 0.00 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| off | 700 | off | off | off | off | 0.05 | 2% | 14% | 0% | 0% | 0% | 0% | 0% | 0% |
| off | 700 | on | off | off | off | 0.07 | 5% | 14% | 0% | 0% | 3% | 0% | 0% | 14% |
| off | 700 | on | 1.5 / off | off | off | 0.15 | 7% | 14% | 7% | 0% | 3% | 0% | 0% | 25% |
| off | 700 | on | 1.5 / 1.0 | off | off | 0.24 | 9% | 17% | 14% | 0% | 3% | 0% | 3% | 29% |
| off | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.24 | 9% | 17% | 14% | 0% | 3% | 0% | 3% | 29% |

## (ii) detector v2 only

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 20 px | off | off | off | off | off | 0.07 | 1% | 3% | 0% | 3% | 0% | 0% | 3% | 0% |
| 0.95 / 20 px | off | off | off | off | off | 0.20 | 3% | 3% | 0% | 7% | 0% | 0% | 3% | 7% |
| 0.9 / 20 px | off | off | off | off | off | 0.32 | 5% | 3% | 0% | 7% | 0% | 0% | 8% | 14% |
| 0.8 / 20 px | off | off | off | off | off | 0.37 | 5% | 3% | 0% | 7% | 0% | 0% | 8% | 14% |
| 0.7 / 20 px | off | off | off | off | off | 0.49 | 6% | 3% | 0% | 7% | 3% | 0% | 11% | 18% |
| 0.5 / 20 px | off | off | off | off | off | 0.59 | 11% | 7% | 3% | 10% | 7% | 0% | 16% | 36% |

## (iii) everything

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.31 | 10% | 17% | 14% | 3% | 3% | 0% | 5% | 29% |
| 0.95 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.44 | 11% | 17% | 14% | 7% | 3% | 0% | 5% | 32% |
| 0.9 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.56 | 13% | 17% | 14% | 7% | 3% | 0% | 11% | 39% |
| 0.8 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.61 | 13% | 17% | 14% | 7% | 3% | 0% | 11% | 39% |
| 0.7 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.73 | 13% | 17% | 14% | 7% | 3% | 0% | 13% | 39% |
| 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.83 | 17% | 21% | 17% | 10% | 7% | 0% | 18% | 43% |

## Chosen point (per type)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.56 | 13% | 17% | 14% | 7% | 3% | 0% | 11% | 39% |

# Held-out trade-off: recall vs false alarms (parser v2B, erasure v2)

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

## (ii) detector both only

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.98 / 40 px + v3 0.98 / 20 px | off | off | off | off | off | 0.00 | 40% | 38% | 17% | 55% | 41% | 0% | 47% | 82% |
| v2 0.95 / 40 px + v3 0.95 / 20 px | off | off | off | off | off | 0.02 | 41% | 38% | 17% | 55% | 41% | 0% | 53% | 86% |
| v2 0.9 / 40 px + v3 0.9 / 20 px | off | off | off | off | off | 0.07 | 44% | 41% | 21% | 55% | 45% | 6% | 55% | 86% |
| v2 0.8 / 40 px + v3 0.8 / 20 px | off | off | off | off | off | 0.10 | 49% | 48% | 28% | 59% | 55% | 6% | 58% | 89% |
| v2 0.7 / 40 px + v3 0.7 / 20 px | off | off | off | off | off | 0.14 | 52% | 52% | 38% | 59% | 62% | 6% | 58% | 89% |
| v2 0.5 / 40 px + v3 0.5 / 20 px | off | off | off | off | off | 0.17 | 58% | 69% | 41% | 62% | 72% | 6% | 66% | 89% |

## (iii) everything

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.98 / 40 px + v3 0.98 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.24 | 45% | 52% | 31% | 55% | 45% | 0% | 50% | 82% |
| v2 0.95 / 40 px + v3 0.95 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.25 | 46% | 52% | 31% | 55% | 45% | 0% | 55% | 86% |
| v2 0.9 / 40 px + v3 0.9 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.31 | 49% | 55% | 34% | 55% | 48% | 6% | 58% | 86% |
| v2 0.8 / 40 px + v3 0.8 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.34 | 53% | 62% | 38% | 59% | 59% | 6% | 58% | 89% |
| v2 0.7 / 40 px + v3 0.7 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.37 | 55% | 66% | 45% | 59% | 66% | 6% | 58% | 89% |
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.41 | 59% | 76% | 45% | 62% | 72% | 6% | 66% | 89% |

## Chosen point (per type)

| detector (threshold / min blob) | C8 min area | R5 | ink: R5 pairs / other interior | R6 | R7 | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 0.5 / 40 px + v3 0.5 / 20 px | 700 | on | 1.5 / 1.0 | x1.2 | off | 0.41 | 59% | 76% | 45% | 62% | 72% | 6% | 66% | 89% |

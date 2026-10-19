# Held-out trade-off: recall vs false alarms (parser v2B, detector v2, erasure v2)

59 intact + 200 omissions (image_eval's held-out cases). Chosen setting: 0.9 / 20 px detector, C8 >= 200.0, R5, ink < 3.0 (relevant doors). Figure: outputs/stage2/check_curve.png.

## (i) rules + ink

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 2756 | off | off | 0.00 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| off | 200 | off | off | 0.15 | 4% | 24% | 0% | 3% | 0% | 0% | 0% | 4% |
| off | 200 | on | off | 0.17 | 7% | 24% | 0% | 3% | 3% | 0% | 0% | 18% |
| off | 200 | on | 0.5 (relevant doors) | 0.25 | 8% | 24% | 7% | 3% | 3% | 0% | 0% | 21% |
| off | 200 | on | 3.0 (relevant doors) | 0.32 | 11% | 28% | 10% | 3% | 7% | 0% | 0% | 29% |
| off | 200 | on | 3.0 (relevant doors) | 0.32 | 11% | 28% | 10% | 3% | 7% | 0% | 0% | 29% |

## (ii) detector v2 only

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 20 px | off | off | off | 0.07 | 1% | 3% | 0% | 3% | 0% | 0% | 3% | 0% |
| 0.95 / 20 px | off | off | off | 0.20 | 3% | 3% | 0% | 7% | 0% | 0% | 3% | 7% |
| 0.9 / 20 px | off | off | off | 0.32 | 5% | 3% | 0% | 7% | 0% | 0% | 8% | 14% |
| 0.8 / 20 px | off | off | off | 0.37 | 5% | 3% | 0% | 7% | 0% | 0% | 8% | 14% |
| 0.7 / 20 px | off | off | off | 0.49 | 6% | 3% | 0% | 7% | 3% | 0% | 11% | 18% |
| 0.5 / 20 px | off | off | off | 0.59 | 11% | 7% | 3% | 10% | 7% | 0% | 16% | 36% |

## (iii) everything

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 20 px | 200 | on | 3.0 (relevant doors) | 0.39 | 12% | 28% | 10% | 7% | 7% | 0% | 3% | 29% |
| 0.95 / 20 px | 200 | on | 3.0 (relevant doors) | 0.53 | 13% | 28% | 10% | 10% | 7% | 0% | 3% | 32% |
| 0.9 / 20 px | 200 | on | 3.0 (relevant doors) | 0.64 | 15% | 28% | 10% | 10% | 7% | 0% | 8% | 39% |
| 0.8 / 20 px | 200 | on | 3.0 (relevant doors) | 0.69 | 15% | 28% | 10% | 10% | 7% | 0% | 8% | 39% |
| 0.7 / 20 px | 200 | on | 3.0 (relevant doors) | 0.81 | 15% | 28% | 10% | 10% | 7% | 0% | 11% | 39% |
| 0.5 / 20 px | 200 | on | 3.0 (relevant doors) | 0.92 | 19% | 31% | 14% | 14% | 10% | 0% | 16% | 46% |

## Chosen point (per type)

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 200 | on | 3.0 (relevant doors) | 0.64 | 15% | 28% | 10% | 10% | 7% | 0% | 8% | 39% |

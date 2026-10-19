# CHECK re-tuned on parsed TUNING drawings (parser v2B, erasure v2)

40 intact + 160 omissions on tuning drawings; budget 0.4 false alarms per intact drawing for the whole CHECK. Found = a proposal overlaps the omission.

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 700 | on | 1.5 | 0.40 | 18% | 22% | 43% | 0% | 22% | 0% | 13% | 27% |

Previous setting (0.9 / 40 px, C8 >= 2756 px, R5, no ink check) on the same drawings:

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 40 px | 2756 | on | off | 0.68 | 21% | 17% | 13% | 30% | 4% | 33% | 19% | 32% |

## Best within budget, top 10

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 700 | on | 1.5 | 0.40 | 18% | 22% | 43% | 0% | 22% | 0% | 13% | 27% |
| off | 1000 | on | 1.5 | 0.40 | 18% | 22% | 43% | 0% | 22% | 0% | 13% | 23% |
| off | 1500 | on | 1.5 | 0.40 | 17% | 17% | 43% | 0% | 22% | 0% | 13% | 23% |
| off | 2756 | on | 1.5 | 0.40 | 16% | 13% | 43% | 0% | 22% | 0% | 13% | 23% |
| off | off | on | 1.5 | 0.40 | 16% | 9% | 43% | 0% | 22% | 0% | 13% | 23% |
| off | 700 | off | 1.5 | 0.33 | 15% | 22% | 39% | 0% | 17% | 0% | 6% | 23% |
| off | 400 | off | 1.5 | 0.38 | 15% | 22% | 39% | 0% | 17% | 0% | 6% | 23% |
| off | 700 | on | 1.0 | 0.40 | 15% | 22% | 35% | 0% | 17% | 0% | 10% | 23% |
| off | 1000 | off | 1.5 | 0.33 | 15% | 22% | 39% | 0% | 17% | 0% | 6% | 18% |
| off | 1000 | on | 1.0 | 0.40 | 15% | 22% | 35% | 0% | 17% | 0% | 10% | 18% |

## What each source adds (best within budget with only that source on)

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (detector only) 0.98 / 40 px | off | off | off | 0.38 | 8% | 0% | 0% | 22% | 4% | 13% | 6% | 9% |
| (C8 only) off | 200 | off | off | 0.12 | 4% | 22% | 0% | 0% | 0% | 0% | 3% | 5% |
| (ink only) off | off | off | 1.5 | 0.33 | 13% | 9% | 39% | 0% | 17% | 0% | 6% | 18% |

# CHECK re-tuned on parsed TUNING drawings (parser v2B, erasure v2)

40 intact + 160 omissions on tuning drawings; budget 0.4 false alarms per intact drawing for the whole CHECK. Found = a proposal overlaps the omission.

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 160 px | 700 | on | 1.5 | 0.40 | 18% | 22% | 39% | 0% | 22% | 0% | 13% | 27% |

Previous setting (0.9 / 40 px, C8 >= 2756 px, R5, no ink check) on the same drawings:

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 2756 | on | off | 0.07 | 4% | 4% | 4% | 0% | 4% | 0% | 6% | 9% |

## Best within budget, top 10

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.98 / 160 px | 700 | on | 1.5 | 0.40 | 18% | 22% | 39% | 0% | 22% | 0% | 13% | 27% |
| off | 700 | on | 1.5 | 0.40 | 18% | 22% | 39% | 0% | 22% | 0% | 13% | 27% |
| 0.98 / 160 px | 1000 | on | 1.5 | 0.40 | 17% | 22% | 39% | 0% | 22% | 0% | 13% | 23% |
| off | 1000 | on | 1.5 | 0.40 | 17% | 22% | 39% | 0% | 22% | 0% | 13% | 23% |
| 0.98 / 160 px | 1500 | on | 1.5 | 0.40 | 16% | 17% | 39% | 0% | 22% | 0% | 13% | 23% |
| off | 1500 | on | 1.5 | 0.40 | 16% | 17% | 39% | 0% | 22% | 0% | 13% | 23% |
| 0.98 / 160 px | 2756 | on | 1.5 | 0.40 | 16% | 13% | 39% | 0% | 22% | 0% | 13% | 23% |
| off | 2756 | on | 1.5 | 0.40 | 16% | 13% | 39% | 0% | 22% | 0% | 13% | 23% |
| 0.95 / 20 px | 700 | off | 0.5 | 0.40 | 15% | 22% | 26% | 13% | 13% | 0% | 6% | 27% |
| 0.95 / 40 px | 700 | off | 1.5 | 0.40 | 15% | 22% | 35% | 0% | 17% | 0% | 10% | 23% |

## What each source adds (best within budget with only that source on)

| detector (threshold / min blob) | C8 min area | R5 | ink threshold | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (detector only) 0.5 / 40 px | off | off | off | 0.30 | 9% | 0% | 4% | 17% | 4% | 0% | 13% | 27% |
| (C8 only) off | 200 | off | off | 0.12 | 4% | 22% | 0% | 0% | 0% | 0% | 3% | 5% |
| (ink only) off | off | off | 1.5 | 0.33 | 12% | 9% | 35% | 0% | 17% | 0% | 6% | 18% |

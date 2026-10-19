# CHECK tuned on parsed TUNING drawings (parser v2B, detector v2, erasure v2)

40 intact + 160 omissions on tuning drawings. Budget 0.6 false alarms per intact drawing in total; rules + ink chosen first within 0.35, then the detector inside what is left (user, 2026-10-19). Found = a proposal overlaps the omission.

## Chosen

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 200 | on | 3.0 (relevant doors) | 0.57 | 19% | 30% | 35% | 17% | 4% | 0% | 16% | 32% |

Stage 1 (rules + ink, detector off; budget 0.35), top 8:

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 200 | on | 3.0 (relevant doors) | 0.35 | 15% | 30% | 35% | 0% | 4% | 0% | 10% | 23% |
| off | 200 | on | 2.0 (relevant doors) | 0.30 | 14% | 30% | 35% | 0% | 4% | 0% | 10% | 18% |
| off | 400 | on | 3.0 (relevant doors) | 0.28 | 13% | 22% | 35% | 0% | 4% | 0% | 10% | 23% |
| off | 200 | on | 1.5 (relevant doors) | 0.25 | 13% | 26% | 35% | 0% | 4% | 0% | 10% | 18% |
| off | 700 | on | 3.0 (relevant doors) | 0.23 | 13% | 22% | 35% | 0% | 4% | 0% | 6% | 23% |
| off | 400 | on | 2.0 (relevant doors) | 0.23 | 13% | 22% | 35% | 0% | 4% | 0% | 10% | 18% |
| off | 700 | on | 2.0 (relevant doors) | 0.17 | 12% | 22% | 35% | 0% | 4% | 0% | 6% | 18% |
| off | 1000 | on | 3.0 (relevant doors) | 0.23 | 12% | 22% | 35% | 0% | 4% | 0% | 6% | 18% |

Stage 2 (detector added to the chosen rules; budget 0.6), top 8:

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 / 20 px | 200 | on | 3.0 (relevant doors) | 0.57 | 19% | 30% | 35% | 17% | 4% | 0% | 16% | 32% |
| 0.95 / 20 px | 200 | on | 3.0 (relevant doors) | 0.47 | 19% | 30% | 35% | 13% | 4% | 0% | 16% | 32% |
| 0.8 / 40 px | 200 | on | 3.0 (relevant doors) | 0.60 | 17% | 30% | 35% | 4% | 4% | 0% | 16% | 32% |
| 0.5 / 80 px | 200 | on | 3.0 (relevant doors) | 0.60 | 17% | 30% | 35% | 0% | 4% | 0% | 13% | 36% |
| 0.98 / 20 px | 200 | on | 3.0 (relevant doors) | 0.45 | 17% | 30% | 35% | 9% | 4% | 0% | 16% | 23% |
| 0.8 / 80 px | 200 | on | 3.0 (relevant doors) | 0.45 | 16% | 30% | 35% | 0% | 4% | 0% | 13% | 32% |
| 0.9 / 40 px | 200 | on | 3.0 (relevant doors) | 0.47 | 16% | 30% | 35% | 0% | 4% | 0% | 13% | 32% |
| 0.7 / 80 px | 200 | on | 3.0 (relevant doors) | 0.55 | 16% | 30% | 35% | 0% | 4% | 0% | 13% | 32% |

## Door-ink check before / after the restriction (ink alone, threshold 3.0, guard on)

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | off | off | 3.0 (all doors) | 0.62 | 16% | 22% | 39% | 0% | 17% | 0% | 13% | 23% |
| off | off | off | 3.0 (relevant doors) | 0.15 | 8% | 9% | 35% | 0% | 0% | 0% | 0% | 9% |

## Detector v2 alone (best within 0.15 / 0.25 / 0.35 false alarms)

| detector (threshold / min blob) | C8 min area | R5 | ink check | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.95 / 20 px | off | off | off | 0.12 | 5% | 0% | 0% | 13% | 0% | 0% | 6% | 14% |
| 0.9 / 20 px | off | off | off | 0.23 | 5% | 0% | 0% | 17% | 0% | 0% | 6% | 14% |
| 0.5 / 40 px | off | off | off | 0.30 | 9% | 0% | 4% | 17% | 4% | 0% | 13% | 27% |

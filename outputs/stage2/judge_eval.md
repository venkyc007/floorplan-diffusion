# Generous flagger + judge vs the current final CHECK (held-out drawings)

59 intact + 200 omissions (image_eval's held-out cases). Judge settings: configs/stage2/judge.yaml (tuned on tuning drawings only).

| CHECK | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| current final (ink tier 2 off) | 0.32 | 58% | 72% | 41% | 62% | 72% | 6% | 66% | 89% |
| generous (no judge) | 1.46 | 70% | 93% | 69% | 72% | 79% | 11% | 71% | 93% |
| generous + judge | 0.42 | 34% | 28% | 10% | 28% | 41% | 11% | 55% | 68% |

**Verdict: the current final stays** (rule: higher held-out mean recall at a false-alarm level no more than 0.1 above the current final).

VERDICT_WIN=False

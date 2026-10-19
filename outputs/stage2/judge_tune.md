# Generous flagger + judge: tuning (parsed TUNING drawings)

40 intact + 160 omissions. Generous setting = best mean recall with 1.2-2.0 false alarms per intact drawing; judge threshold for <= 0.4.

Generous setting: {'det': (0.3, 10), 'det3': (0.3, 10), 'c8': 0, 'r5': True, 'ink': 1.5, 'r6': 1.0}. Logistic coefficients (intercept, judge, ink, is_door): [-0.105, 1.453, -2.702, 0.412]; threshold 0.5. Flags used for the fit: 588 (174 true).

| CHECK | false alarms / intact | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| current final (ink tier 2 off) | 0.20 | 56% | 43% | 74% | 70% | 57% | 0% | 55% | 95% |
| generous (no judge) | 1.20 | 66% | 65% | 83% | 78% | 78% | 0% | 65% | 95% |
| generous + judge | 0.25 | 34% | 9% | 30% | 30% | 43% | 0% | 35% | 86% |

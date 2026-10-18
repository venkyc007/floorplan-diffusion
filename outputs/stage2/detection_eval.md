# Missing-component detection on whole plans (CHECK step)

300 typed val plans: 100 intact (val plans 200-299 of the fixed order) and 200 with ONE simulated omission each (seed 4, kinds in turn). Detected = a proposal box overlapping the omitted pixels; type-correct = that proposal has the right type. The detector setting was chosen on these same plans.

**Chosen detector setting: missing probability >= 0.9, blobs >= 40 px** (0.50 false alarms per intact plan, mean recall 67%).

| method | false alarms / intact plan | mean recall | door | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|
| rules only (all rules) | 0.52 | 7% | 9% (9%) | 0% (0%) | 3% (0%) | 4% (0%) | 6% (3%) | 19% (3%) |
| detector only | 0.50 | 67% | 46% (40%) | 71% (71%) | 74% (69%) | 68% (68%) | 43% (26%) | 97% (97%) |
| both (detector + C8/C17/C6/C12) | 0.85 | 67% | 49% (43%) | 71% (71%) | 74% (66%) | 68% (64%) | 46% (29%) | 97% (78%) |

Cells: detected (type-correct).

## Detector sweep (false alarms per intact plan / mean recall)

| threshold | blob >= 5 | blob >= 10 | blob >= 20 | blob >= 40 | blob >= 80 |
|---|---|---|---|---|---|
| 0.3 | 2.95 / 86% | 2.43 / 86% | 2.02 / 83% | 1.37 / 75% | 0.89 / 60% |
| 0.4 | 2.58 / 86% | 2.15 / 86% | 1.71 / 81% | 1.19 / 75% | 0.82 / 59% |
| 0.5 | 2.20 / 85% | 1.91 / 84% | 1.47 / 81% | 1.08 / 74% | 0.73 / 58% |
| 0.6 | 1.93 / 84% | 1.64 / 82% | 1.29 / 80% | 0.96 / 72% | 0.60 / 58% |
| 0.7 | 1.68 / 83% | 1.42 / 82% | 1.15 / 79% | 0.84 / 70% | 0.49 / 56% |
| 0.8 | 1.50 / 81% | 1.22 / 79% | 0.89 / 77% | 0.66 / 68% | 0.38 / 54% |
| 0.9 | 1.01 / 78% | 0.82 / 77% | 0.65 / 74% | 0.50 / 67% | 0.25 / 50% |
| 0.95 | 0.76 / 77% | 0.63 / 76% | 0.53 / 73% | 0.38 / 62% | 0.21 / 48% |

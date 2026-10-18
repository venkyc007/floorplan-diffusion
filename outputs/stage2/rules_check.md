# Rule-based detector: check on simulated omissions

Thresholds: `configs/stage2/detect_rules.yaml` (tuned on 200 intact val plans, <= 0.5 false alarms per plan). 287 val plans with ONE simulated omission each (seed 1, omission kinds in turn); a detection = a candidate box (after thresholds and merging) overlapping the omitted pixels; type-correct = that candidate also has the right type.

**False alarms on 100 other intact val plans: 0.38 candidates per plan** (by first rule: C14 0.06, C17 0.05, C18 0.04, C13 0.03, C11 0.03, C12 0.03, C3 0.02, C4 0.02, C16 0.02, C6 0.02, C7 0.02, C15 0.01, R3 0.01, R1 0.01, C8 0.01).

**Overall: detected 7%, type-correct 3%; candidates per omitted plan 0.45.**

| kind | n | detected | type-correct |
|---|---|---|---|
| door | 50.0 | 10% | 2% |
| fixture | 49.0 | 4% | 0% |
| railing | 42.0 | 2% | 0% |
| room | 46.0 | 17% | 17% |
| wall | 50.0 | 4% | 2% |
| window | 50.0 | 4% | 0% |

| location | n | detected | type-correct |
|---|---|---|---|
| exterior | 148.0 | 6% | 5% |
| interior | 139.0 | 8% | 2% |

| size_bin | n | detected | type-correct |
|---|---|---|---|
| large | 54.0 | 15% | 15% |
| medium | 79.0 | 8% | 0% |
| small | 154.0 | 4% | 1% |

Thresholds and false alarms per rule on the 200 tuning plans:

| rule | threshold (score) | false alarms / plan | raw candidates / plan |
|---|---|---|---|
| C1 | 1e+12 | 0.000 | 5.84 |
| C3 | 18 | 0.030 | 3.29 |
| C4 | 276 | 0.030 | 9.19 |
| C6 | 230 | 0.030 | 0.46 |
| C7 | 960 | 0.030 | 6.37 |
| C9 | 248 | 0.030 | 2.13 |
| C10 | 13 | 0.030 | 4.57 |
| C11 | 138 | 0.025 | 3.08 |
| C12 | 364 | 0.030 | 6.17 |
| C13 | 116 | 0.030 | 0.89 |
| C14 | 14 | 0.020 | 0.90 |
| C15 | 120 | 0.030 | 0.23 |
| C16 | 143 | 0.025 | 0.21 |
| C18 | 273 | 0.025 | 1.15 |
| C8 | 8625 | 0.035 | 1.95 |
| C17 | 957 | 0.035 | 0.10 |
| R1 | 1.198e+04 | 0.035 | 0.16 |
| R3 | 91 | 0.025 | 0.10 |
| R4 | 0 | 0.005 | 0.01 |

# Rule-based detector: check on simulated omissions

Thresholds: `configs/stage2/detect_rules.yaml` (tuned on 200 intact val plans, <= 0.5 false alarms per plan). 287 val plans with ONE simulated omission each (seed 1, omission kinds in turn); a detection = a candidate box (after thresholds and merging) overlapping the omitted pixels; type-correct = that candidate also has the right type.

**False alarms on 100 other intact val plans: 0.52 candidates per plan** (by first rule: C17 0.15, C12 0.12, C6 0.11, C11 0.04, C18 0.03, C14 0.03, C16 0.02, C8 0.02).

**Overall: detected 8%, type-correct 3%; candidates per omitted plan 0.43.**

| kind | n | detected | type-correct |
|---|---|---|---|
| door | 50.0 | 8% | 6% |
| fixture | 49.0 | 8% | 4% |
| railing | 42.0 | 2% | 0% |
| room | 46.0 | 15% | 2% |
| wall | 50.0 | 4% | 4% |
| window | 50.0 | 10% | 0% |

| location | n | detected | type-correct |
|---|---|---|---|
| exterior | 148.0 | 5% | 1% |
| interior | 139.0 | 12% | 5% |

| size_bin | n | detected | type-correct |
|---|---|---|---|
| large | 54.0 | 11% | 2% |
| medium | 79.0 | 6% | 0% |
| small | 154.0 | 8% | 5% |

Thresholds and false alarms per rule on the 200 tuning plans:

| rule | threshold (score) | false alarms / plan | raw candidates / plan |
|---|---|---|---|
| C1 | 1e+12 | 0.000 | 5.84 |
| C3 | 88 | 0.005 | 3.29 |
| C4 | 655 | 0.005 | 9.19 |
| C6 | 84 | 0.100 | 0.46 |
| C7 | 3341 | 0.005 | 6.37 |
| C9 | 350 | 0.005 | 2.13 |
| C10 | 1e+12 | 0.000 | 4.57 |
| C11 | 260 | 0.005 | 3.08 |
| C12 | 244 | 0.100 | 6.17 |
| C13 | 252 | 0.005 | 0.89 |
| C14 | 18 | 0.010 | 0.90 |
| C15 | 306 | 0.010 | 0.23 |
| C16 | 163 | 0.010 | 0.21 |
| C18 | 516 | 0.010 | 1.15 |
| C8 | 2756 | 0.100 | 0.45 |
| C17 | 37 | 0.100 | 0.10 |
| R1 | 1.816e+04 | 0.010 | 0.16 |
| R3 | 224 | 0.010 | 0.10 |
| R4 | 0 | 0.005 | 0.01 |

## Diagnostic 3: false-alarm budget re-split (C8, C17, C6, C12: 0.1 per plan each)

The tables at the top use these thresholds. Before the re-split (equal ~0.03 per rule, and C8 walking only from the largest walkable area) the rules detected 7% (type-correct 3%) at 0.38 false alarms per plan.

| rule | threshold | false alarms / plan (tuning) |
|---|---|---|
| C8 | 2756 | 0.1 |
| C17 | 37 | 0.1 |
| C6 | 84 | 0.1 |
| C12 | 244 | 0.1 |
| C1 | 1e+12 | 0.0 |
| C3 | 88 | 0.005 |
| C4 | 655 | 0.005 |
| C7 | 3341 | 0.005 |
| C9 | 350 | 0.005 |
| C10 | 1e+12 | 0.0 |
| C11 | 260 | 0.005 |
| C13 | 252 | 0.005 |
| C14 | 18 | 0.01 |
| C15 | 306 | 0.01 |
| C16 | 163 | 0.01 |
| C18 | 516 | 0.01 |
| R1 | 1.816e+04 | 0.01 |
| R3 | 224 | 0.01 |
| R4 | 0 | 0.005 |

## Diagnostic 2: the TA case (a room's ONLY door replaced by wall)

100 val plans. C8 recall: **raw 85%**, with the re-split thresholds 13% (with the old thresholds 1%). All 15 raw misses are rooms that stay reachable through an open passage (no door), so C8 is right not to fire; among rooms that really become cut off, raw C8 finds 100%. Fix applied: C8 now walks from the main flat AND from every walkable area with an exterior door (plans with several flats/floors); raw C8 false alarms on intact plans fell from 1.96 to 0.45 per plan.

## Diagnostic 1: rule-visibility ceiling

Not run: the job was stopped by Claude Code (laptop low on memory). Re-run with `python -m src.stage2.detect.rules_diag --zip <completion_val.zip> --part ceiling`.

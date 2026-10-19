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

## Held-out re-score (honest numbers; user, chunk 6)

Setting fixed at threshold 0.9 / blobs >= 40 px. Val has only 371 typed plans; 159 were NOT used to choose the detector setting: 59 stay intact, each of the other 100 gets two independent omissions of different kinds (200 cases). Caveat: the rule thresholds were tuned on intact val plans 0-199, which include most of these plans, so the rule columns are slightly optimistic.

| method | false alarms / intact plan | mean recall | door | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|
| rules only (all rules) | 0.27 | 8% | 0% (0%) | 0% (0%) | 6% (0%) | 9% (0%) | 15% (5%) | 18% (6%) |
| detector only | 0.29 | 61% | 28% (28%) | 73% (73%) | 83% (74%) | 48% (48%) | 38% (33%) | 97% (97%) |
| detector + C8 | 0.31 | 62% | 28% (28%) | 73% (73%) | 83% (74%) | 52% (48%) | 38% (33%) | 97% (97%) |
| detector + C8/C17/C6/C12 | 0.47 | 64% | 28% (28%) | 73% (73%) | 83% (74%) | 57% (48%) | 46% (33%) | 100% (88%) |

**Final CHECK setting: detector + C8** (rule: detector + C8 if <= 0.6 false alarms per plan, else detector only). The detector already takes the rule violation maps as input, so it is itself the rules + learned combination.

## Per-type detector thresholds (user, chunk 7)

Tuned greedily on the tuning split within 0.4 false alarms per intact plan (incl. C8), doors weighted x2, railing and fixtures x0.5 (blobs of each type's own probability; min blob 40 px, fixtures 15 px). Tuning-split false alarms: 0.35. Thresholds: wall 0.7, door 0.95, window 0.98, railing 0.9, room 0.98, toilet 0.98, sink 0.85, bath_shower 0.995, appliance 0.995.

Held-out (the same 159 plans as above), both with C8:

| setting | false alarms / intact plan | mean recall | door | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|
| single threshold 0.9 (per-type blobs) | 0.31 | 61% | 22% (22%) | 73% (73%) | 80% (74%) | 52% (48%) | 41% (36%) | 97% (97%) |
| per-type thresholds | 0.20 | 58% | 17% (17%) | 67% (67%) | 77% (74%) | 52% (48%) | 38% (36%) | 94% (94%) |

**Per-type thresholds: tried and REJECTED (user, chunk 7 follow-up).** Held-out mean recall fell 61% -> 58% and doors 22% -> 17%; the greedy search started every type at 0.995 and stopped early with budget left (0.35 of 0.4 on tuning, 0.20 held-out), leaving doors stricter, not looser. The pipeline keeps the single threshold 0.9 / 40 px + C8.

## Held-out on DRAWINGS with realistic erasure (overnight, image level)

1 intact drawings + 2 omissions on 1 other held-out drawings (kinds: door (a) 1, door (b) 1, window 0, wall 0, railing 0, fixture 0, room 0). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |
| detector + C8 | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |
| detector + C8 + R5 (final) | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |

Scope subsets (detector + C8 + R5 (final)): door (a) where the closure cuts a room off: 0 cases, found -; door (b) between a bedroom/bathroom and a closed-pair neighbour: 0 cases, found -.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS with realistic erasure (overnight, image level)

1 intact drawings + 2 omissions on 1 other held-out drawings (kinds: door (a) 1, door (b) 1, window 0, wall 0, railing 0, fixture 0, room 0). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |
| detector + C8 | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |
| detector + C8 + R5 (final) | 1.00 | 100% | 100% (0%) | 100% (0%) | - | - | - | - | - |

Scope subsets (detector + C8 + R5 (final)): door (a) where the closure cuts a room off: 0 cases, found -; door (b) between a bedroom/bathroom and a closed-pair neighbour: 0 cases, found -.
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 100%, door (b) 100%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS with realistic erasure (overnight, image level)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.85 | 10% | 7% (0%) | 7% (0%) | 10% (0%) | 0% (0%) | 11% (6%) | 13% (11%) | 21% (0%) |
| detector + C8 | 0.85 | 10% | 7% (0%) | 7% (0%) | 10% (0%) | 0% (0%) | 11% (6%) | 13% (11%) | 21% (0%) |
| detector + C8 + R5 (final) | 0.95 | 12% | 7% (0%) | 7% (0%) | 10% (0%) | 3% (0%) | 11% (6%) | 13% (11%) | 32% (0%) |

Scope subsets (detector + C8 + R5 (final)): door (a) where the closure cuts a room off: 15 cases, found 7% (0%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 0% (0%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 55%, door (b) 85%, window 77%, wall 26%, railing 17%, fixture 8%, room 23%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS, re-run with erasure v2 (whole symbol removed; 2026-10-19)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.85 | 11% | 3% (0%) | 3% (0%) | 21% (21%) | 0% (0%) | 11% (6%) | 18% (16%) | 21% (0%) |
| detector + C8 | 0.85 | 11% | 3% (0%) | 3% (0%) | 21% (21%) | 0% (0%) | 11% (6%) | 18% (16%) | 21% (0%) |
| detector + C8 + R5 (final) | 0.95 | 13% | 3% (0%) | 3% (0%) | 21% (21%) | 3% (0%) | 11% (6%) | 18% (16%) | 32% (0%) |

Scope subsets (detector + C8 + R5 (final)): door (a) where the closure cuts a room off: 15 cases, found 0% (0%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 0% (0%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 23%, door (b) 81%, window 45%, wall 26%, railing 17%, fixture 4%, room 23%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS: parser v2B + re-tuned CHECK (detector off, C8 >= 700 px, R5, door ink; 2026-10-19)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.00 | 0% | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) |
| detector + C8 | 0.05 | 2% | 14% (14%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) |
| detector + C8 + R5 | 0.07 | 5% | 14% (14%) | 0% (0%) | 0% (0%) | 3% (0%) | 0% (0%) | 0% (0%) | 14% (0%) |
| detector + C8 + R5 + ink (final) | 0.42 | 12% | 24% (24%) | 21% (21%) | 0% (0%) | 3% (0%) | 0% (0%) | 3% (0%) | 32% (0%) |

Scope subsets (detector + C8 + R5 + ink (final)): door (a) where the closure cuts a room off: 15 cases, found 33% (33%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 36% (36%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 30%, door (b) 80%, window 42%, wall 27%, railing 16%, fixture 4%, room 24%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS: detector v2 + CHECK re-tuned with it (detector 0.98 / 160 px, C8 >= 700 px, R5, door ink + guard; 2026-10-19)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.00 | 0% | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) |
| detector + C8 | 0.05 | 2% | 14% (14%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) | 0% (0%) |
| detector + C8 + R5 | 0.07 | 5% | 14% (14%) | 0% (0%) | 0% (0%) | 3% (0%) | 0% (0%) | 0% (0%) | 14% (0%) |
| detector + C8 + R5 + ink (final) | 0.29 | 12% | 24% (24%) | 21% (21%) | 0% (0%) | 3% (0%) | 0% (0%) | 3% (0%) | 32% (0%) |

Scope subsets (detector + C8 + R5 + ink (final)): door (a) where the closure cuts a room off: 15 cases, found 33% (33%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 36% (36%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 30%, door (b) 80%, window 42%, wall 27%, railing 16%, fixture 4%, room 24%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS: budget 0.6 (detector v2 0.9 / 20 px, C8 >= 200 px, R5, ink < 3.0 on R5 pairs; 2026-10-19)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.32 | 5% | 3% (0%) | 0% (0%) | 7% (7%) | 0% (0%) | 0% (0%) | 8% (8%) | 14% (11%) |
| detector + C8 | 0.47 | 9% | 24% (24%) | 0% (0%) | 10% (7%) | 0% (0%) | 0% (0%) | 8% (8%) | 18% (11%) |
| detector + C8 + R5 | 0.49 | 11% | 24% (24%) | 0% (0%) | 10% (7%) | 3% (0%) | 0% (0%) | 8% (8%) | 29% (7%) |
| detector + C8 + R5 + ink (final) | 0.64 | 15% | 28% (28%) | 10% (10%) | 10% (7%) | 7% (0%) | 0% (0%) | 8% (8%) | 39% (7%) |

Scope subsets (detector + C8 + R5 + ink (final)): door (a) where the closure cuts a room off: 15 cases, found 53% (53%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 21% (21%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 30%, door (b) 80%, window 42%, wall 27%, railing 16%, fixture 4%, room 24%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS: + R6 / R7 / ink tiers (det v2 0.9 / 20, C8 >= 700, R5, ink 1.5 / 1.0, R6 x1.2; 2026-10-19; the final row has no R6, which adds nothing here, see check_curve.md)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.32 | 5% | 3% (0%) | 0% (0%) | 7% (7%) | 0% (0%) | 0% (0%) | 8% (8%) | 14% (11%) |
| detector + C8 | 0.37 | 6% | 14% (14%) | 0% (0%) | 7% (7%) | 0% (0%) | 0% (0%) | 8% (8%) | 14% (11%) |
| detector + C8 + R5 | 0.39 | 8% | 14% (14%) | 0% (0%) | 7% (7%) | 3% (0%) | 0% (0%) | 8% (8%) | 25% (7%) |
| detector + C8 + R5 + ink (final) | 0.56 | 13% | 17% (17%) | 14% (14%) | 7% (7%) | 3% (0%) | 0% (0%) | 11% (8%) | 39% (7%) |

Scope subsets (detector + C8 + R5 + ink (final)): door (a) where the closure cuts a room off: 15 cases, found 33% (33%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 29% (29%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 30%, door (b) 80%, window 42%, wall 27%, railing 16%, fixture 4%, room 24%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

## Held-out on DRAWINGS: final CHECK = detectors v2 + v3 + C8 + R5 + ink tiers + R6 (2026-10-19). Caveat: v3 may partly detect the erasure's own traces (see decisions.md)

59 intact drawings + 200 omissions on 100 other held-out drawings (kinds: door (a) 29, door (b) 29, window 29, wall 29, railing 18, fixture 38, room 28). Erasure: src/stage2/data/erasure.py; then the pipeline's PARSE + CHECK (no fill). Found = a proposal overlaps the omission (in brackets: right type).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector | 0.17 | 58% | 69% (55%) | 41% (34%) | 62% (62%) | 72% (45%) | 6% (0%) | 66% (39%) | 89% (89%) |
| detector + C8 | 0.22 | 58% | 72% (66%) | 41% (34%) | 62% (62%) | 72% (45%) | 6% (0%) | 66% (39%) | 89% (89%) |
| detector + C8 + R5 | 0.24 | 58% | 72% (66%) | 41% (34%) | 62% (62%) | 72% (45%) | 6% (0%) | 66% (39%) | 89% (79%) |
| detector + C8 + R5 + ink | 0.41 | 59% | 76% (69%) | 45% (38%) | 62% (62%) | 72% (45%) | 6% (0%) | 66% (39%) | 89% (75%) |
| full CHECK (+ R6 + R7, final) | 0.41 | 59% | 76% (69%) | 45% (38%) | 62% (62%) | 72% (45%) | 6% (0%) | 66% (39%) | 89% (75%) |

Scope subsets (full CHECK (+ R6 + R7, final)): door (a) where the closure cuts a room off: 15 cases, found 73% (60%); door (b) between a bedroom/bathroom and a closed-pair neighbour: 14 cases, found 57% (50%).
Parser redraws the removed component (mean share of its pixels the parser still labels with the removed class, room = its walls): door (a) 30%, door (b) 80%, window 42%, wall 27%, railing 16%, fixture 4%, room 24%.
Note: label-level numbers above are on ground-truth label maps; these are on parsed drawings, so parser errors count against detection (and add false alarms).

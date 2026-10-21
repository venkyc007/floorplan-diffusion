# Overnight report 2 (2026-10-19/20)

All evaluation ran as Kaggle kernels. The laptop only ran code, unit tests and smoke tests (<= 3 items, or 20 cases
with 3 sampling steps for Stage 1). Everything is committed locally on `stage2` (not pushed). Every own choice is in
`docs/decisions.md` ("Overnight 2026-10-19/20").

**GPU hours left this week: about 5 h** (estimate from the run logs: this night's GPU session ~1.6 h; Kaggle's
settings page has the exact figure).

## Task 1: Stage 1 final test evaluation (run once): done

- **What was done**
  - `--benchmark val|test` added to `src/stage1/full_eval.py` and `src/stage1/arch_eval.py`; default stays val.
  - `src/stage1/selection.py --benchmark test` scores the saved test predictions with the val metrics. It builds the
    FROZEN size-adaptive hybrid (t0 = 400 / 600 / 700 per size bin, no re-selection), runs the same paired bootstrap
    and writes the val-vs-test table.
  - Test benchmark uploaded as the private dataset `floorplan-test-benchmark` (3 MB, checksum verified in the kernel).
  - Kernels: `floorplan-stage1-test` (GPU T4 x2, sampling only, ~1.5 h) and `floorplan-stage1-test-score` (CPU, scoring).
  - Smoke test: 20 test cases, 3 steps, the whole pipeline end to end.
- **Results: `outputs/stage1/stage1_test.md`.** No metric of any method differs from val by more than 2 points.

  | method | tolerant mIoU val / test | validity % val / test | connectivity error val / test |
  |---|---|---|---|
  | size-adaptive hybrid (final) | 0.842 / **0.846** | 88.0 / **88.0** | 0.011 / **0.011** |
  | matched U-Net | 0.846 / 0.850 | 80.4 / 81.0 | 0.092 / 0.085 |
  | hybrid t0 = 600 | 0.842 / 0.846 | 87.4 / 87.6 | 0.023 / 0.022 |
  | refine-600 (constraint fine-tune) | 0.801 / 0.804 | 83.6 / 84.1 | 0.039 / 0.037 |
  | nearest fill | 0.489 / 0.493 | 36.0 / 36.6 | 0.066 / 0.064 |

  Bootstrap on test gives the same conclusions as on val:
  - size-adaptive vs matched U-Net: clearly better connectivity and validity, slightly worse tolerant mIoU;
  - vs refine-600: better on all three;
  - vs hybrid 600: better connectivity, no clear difference on mIoU / validity overall.

  Per type and size: tables in the same file.
- **Failure:** the first launch stopped at its own safety check. It compared the npz file hash, while the stored
  checksum is over the arrays. Fixed and relaunched (no GPU work was lost). It also queued ~2 h for a T4.
- CLAUDE.md's Stage 1 status is updated. **The test set must not be evaluated again.**

## Task 2: generous flagger + judge: tried, does NOT win (current final stays)

- **Step 0:** ink tier 2 (other interior doors) switched off (check-set case 7).
- **Generous flagger** (tuning drawings, best recall within 1.2-2.0 false alarms): detectors v2 + v3 at 0.3 / 10 px,
  C8 >= 0 px, R5, ink 1.5, R6 x1.0.
- **Judge:** the k2a 16-class U-Net on the masked box (unforced, no sketch prior). Score = mean P(type) in the box
  minus the parse's share. A logistic on [judge, door ink, is_door] was fitted on 588 tuning flags; threshold 0.5
  for <= 0.4 false alarms.
- **Held-out (159 drawings, `outputs/stage2/judge_eval.md`):**

  | CHECK | false alarms / intact | mean recall | door (a) | door (b) | window | wall | fixture | room |
  |---|---|---|---|---|---|---|---|---|
  | current final (ink tier 2 off) | **0.32** | **58%** | 72% | 41% | 62% | 72% | 66% | 89% |
  | generous, no judge | 1.46 | 70% | 93% | 69% | 72% | 79% | 71% | 93% |
  | generous + judge | 0.42 | 34% | 28% | 10% | 28% | 41% | 55% | 68% |

- **Why it fails:** the judge's score hardly separates true omissions from false flags (`outputs/stage2/judge_scores.png`).
  - For every type except doors, the score sits near 0.5 for both true and false flags.
  - For doors it mostly repeats the ink check.
  - The unforced U-Net simply re-draws the parse's context.
- **Decision (your rule):** keep the current final setting; generous + judge is logged as a tried ablation
  (`use_judge` stays off; code and `configs/stage2/judge.yaml` kept for the report).
- **12-drawing check set (final setting, ink tier 2 off):** 4 / 12 correct + 3 partly (automatic 5 / 12). Same score
  as before. Case 7 now has 3 false ink doors instead of 6: the remaining 3 come from tier 1, because that drawing's
  bedroom / bathroom doors have no arcs either. Look at `outputs/stage2/history/check_set.png`, `check_set.md`,
  `check_set_holes.png`.

## Decisions needed

1. **High-recall mode?** The generous flagger alone reaches 70% held-out recall at 1.46 false alarms per intact
   drawing (vs 58% at 0.32). It could be a user-selectable "find more, check more" option in the Gradio app. Off
   unless you want it.
2. **Case 7:** the ink check (tier 1) still fires on drawings whose doors are drawn without arcs. The per-drawing
   guard only triggers when more than half of ALL doors lack ink. A stricter guard (e.g. > 30%) would trade a little
   door recall; not tested.
3. **Stage 1 is closed.** The test numbers are final; `outputs/stage1/stage1_test.md` is ready for the report.

# outputs/stage2 — Stage 2 results (`final_test.md` = the TEST result, run once 2026-10-22; everything else is VALIDATION)

**Data role** says how far each number can be trusted:
**T** = measured on the *tuning* val drawings the settings were chosen on (optimistic);
**H** = *held-out* val drawings (not used to choose that setting, but some held-out drawings informed other choices,
so these are still validation numbers, not test numbers); **H\*** = held-out drawings that were used for the choice
in that file; **L** = label maps / data statistics (no drawings); **fig** = illustration.
"Current" = matches the final configuration (`docs/decisions.md`, top).

## Headline files (current configuration)

| file | what | role |
|---|---|---|
| **`final_test.md`** | **the Stage 2 TEST result (run once)**: FILL arms + constraint ablation, CHECK, end to end, all breakdowns | **TEST** |
| **`test_components.md`** | **second TEST evaluation (user request)**: end to end on single-component omissions, all 371 test drawings, by kind / location / position | **TEST** |
| `parser_v2.md` | PARSE: parser v1 / v2A / v2B, normal and room names erased (v2B chosen) | H\* |
| `check_tune.md` | CHECK: how the final setting was chosen (budget 0.4), contribution of each part | T |
| `detection_eval.md` | CHECK on held-out drawings; the last section is nearest to the final setting | H |
| `b4_val.md` | FILL with the hole given: small vs large holes, current vs B4 models | H\* |
| `e2e_val_batch2_b4.md` / `.png` | END TO END, 12 fresh held-out drawings, final configuration (before the audit fixes) | H |
| `e2e_val_v5.md`, `e2e_val_batch2_v2.md`, `e2e_val_batch3_v2.md` | END TO END re-check after the 2026-10-21 audit fixes (old vs new tables at the end) | H |
| `large_fill_compare_v2*.md` / `.png` | large omissions: three fills with the B4 models, box given (manual) and automatic | H\* |
| `data_stats.md`, `data_overlay.png` | dataset: class shares, scales, the 16 classes | L |
| `scale_diag.md` / `.csv` | what the ~0.84x scale mismatch costs the FILL (hole given, 30 + 30 holes at both scales): small, -2.0 / +0.1 points | H\* (val windows) |
| `pipeline_steps.png`, `pipeline_steps/` | the four step pictures of the final system on the first gallery drawing of each kind (2026-10-21, after the audit fixes) | H, fig |

## Supporting files

| file | what | role |
|---|---|---|
| `constraint_gt_violations.md`, `constraint_weights.md/.yaml`, `gate_g3*.md`, `gate_g4*.md`, `constraint_toys.png` | constraint library gates G1–G4, gradient-size weights | L |
| `detector_compare.md` | detectors v1 / v2 / v3 on the same crops | T |
| `check_curve.md/.png` | recall vs false alarms trade-off (19 Oct setting) | H |
| `check_set_v4.md/.png`, `check_set_holes_v4.png` | 12-drawing visual check set (20 Oct setting) | H |
| `e2e_val_v4.md`, `e2e_val_batch2.md`, `e2e_val_batch3.md` | end-to-end batches before B4 (batch 1 drawings were re-run after fixes, so batch 1 is partly tuned on) | H\* / H |
| `t0_sweep.md`, `fast_vs_full.md`, `eta_tune.md` | sampler choices: t0, fast vs full, guidance strength | H\* / T |
| `fake_erasure.md`, `redraw_rate.md`, `text_dependence.md`, `c8_diag.md`, `gap_rule.md`, `rules_check.md` | diagnostics behind CHECK / PARSE choices | H / T |
| `judge_tune.md`, `judge_eval.md`, `judge_scores.png` | rejected alternative: generous flagger + U-Net judge | T / H |
| `erasure_check.png`, `erasure_large_examples.png`, `omission_examples.png`, `text_erasure_examples.png`, `parser_aug_examples.png`, `sketch_prior_examples.png`, `pipeline_examples_v2.*`, `detection_examples.png` | how omissions, augmentations and the sketch prior look | fig |
| `*.csv` | per-case rows behind the tables above | |

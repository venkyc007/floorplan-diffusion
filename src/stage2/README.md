# src/stage2 — completing real floor plan drawings (PARSE → CHECK → FILL → VERIFY)

Start at **`pipeline/run.py`**: `run_pipeline(drawing)` is the whole method, and its docstring walks through every
step. Settings: `configs/stage2/pipeline.yaml` (checkpoints, parse, fill, sampling) and
`configs/stage2/detector_threshold.yaml` (the final CHECK). Every module has a docstring saying what it is for.

Roles: **method** = used by `run_pipeline` (the shipped system); **training** = how a model was trained;
**data** = dataset build; **evaluation** = how numbers were measured (val, the final test); **experiment** = a
one-off diagnostic or figure behind a decision in `docs/decisions.md` (kept so every number can be reproduced).

| package | method | training / data | evaluation | experiment |
|---|---|---|---|---|
| `data/` | `classes16` (the 16 classes), `rasterise`, `plan_stats`, `erasure`, `erasure_large`, `omission_meta` | `build_cubicasa` (Kaggle CPU build), `cubicasa_svg` (SVG reader), `survey`, `report` | | `erasure_check` |
| `constraints/` | `constraints` (C1–C18 violation maps = losses, detector input, verifier), `ops` (differentiable morphology), `rules` (R1–R4) | | | `toys`, `make_toy_figure` (gate G1) |
| `parse/` | `train.Parser` + `predict_safe` (the network), `augment`, `dataset`, `losses`, `metrics` | `train` (v1), `train_v2` (v2B: text erasure + constraint fine-tune), `text_erase` | `eval_v2` (held-out table), `val_split` (tuning vs held-out ids) | `text_dependence`, `make_aug_examples` |
| `detect/` | `rules` (C8, flag merge), `gaps` (R5), `door_ink`, `symbol_ink` (window ink), `window_runs` (R6), `merged_rooms` (R8), `narrow_gaps` (R7, off), `flag_gap`, `detection_eval` (detector v2 inference), `detector_v3` | `train_detector` (shared network code), `train_detector_v2`, `train_detector_v3`, `make_parsed_set`, `make_crop_set`, `omissions` | `check_tune` (how CHECK was tuned), `image_eval` (omission generator + held-out tables), `omission_mix` (test size mix) | `c8_diag`, `check_curve`, `door_eval`, `fake_erasure`, `gap_tune`, `redraw_eval`, `rules_diag`, `rules_eval`, `judge` (rejected) |
| `complete/` | `models16` (16-class U-Net + diffusion), `sampler` (hybrid sampling + best-of-N verifier), `sketch_prior`, `large_fill` (large holes, B4), `windows` | `train16`, `warm_start` (6 → 16 classes), `data16`, `calibrate` (gradient-size weights), `gates34` | `t0_sweep` (`hole_ious` metric), `b4_val` (`score`, `violations`) | `eta_tune`, `fast_vs_full`, `gate_g2`, `prior_sizes` |
| `pipeline/` | `run` (the method) | | `e2e_val` (end-to-end scoring + val batches) | `check_set`, `large_fill_compare`, `make_examples`, `prior_examples`, `erasure_large_examples` |
| `interface/` | `demo` (the app's Stage 2 tab: inputs, step pictures, scoring) | `make_gallery` (the app's 18 val examples) | | |
| top level | | `package_final` (checkpoint dataset for the test) | `final_test` (the one-shot test), `kaggle_cpu` (CPU evaluation kernels) | |

Data roles: the **test** split is read only by `final_test.py`. Val drawings are split into **tuning** (212; used
to choose settings) and **held-out** (159; used to check them) in `configs/stage2/val_split/`.

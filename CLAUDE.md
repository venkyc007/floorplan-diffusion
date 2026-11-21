# CLAUDE.md

Working context for Claude Code (the AI coding assistant used in this project). Read at the start of every session,
then `docs/decisions.md` (final configuration at the top, choices in date order), `docs/results.md`,
`docs/kaggle_workflow.md`. The human-facing entry point is `README.md`.

## Project

IIT Bombay IE643 course project, solo, team name **One Hot Mencoding**. **Diffusion-based reconstruction of missing
floor plan components** (allotment: an engineer may unknowingly miss a component while drawing; reconstruct it with
diffusion). The allotment's expected deliverables are mapped to the repo in `README.md` ("Deliverables checklist").
Submission (report PDF, slides, code-walkthrough video, demo video) **Sun 25 Oct 2026 23:59 IST**; in-person
presentation later. Favour simple, working solutions.

## Status (updated 2026-10-21 night)

- **Stage 1 done** (tag `stage1-final`); test evaluated once (`outputs/stage1/stage1_test.md`).
- **Stage 2 done; test run once on 2026-10-22** (`outputs/stage2/final_test.md`, summary `docs/results.md` §0): FILL
  diffusion finish 18% -> 52% valid; constraint ablation no effect; end to end 66% found, 12% fixed. Second user-requested evaluation (single-component omissions, all 371 test
  drawings): 41% fixed (`outputs/stage2/test_components.md`). Never run again.
- **2026-10-21 night (branch `final`)**: full audit (104 findings; cheap fixes applied, the rest documented in
  decisions.md), repo reorganised (`src/stage1/`, `configs/stage1/`, `tests/stage1/`, `outputs/stage1/`,
  `kaggle/kernels/stage{1,2}/`, superseded Stage 2 results in `outputs/stage2/history/`), combined Gradio app
  (Stage 1 + Stage 2 tabs), final test fixed and prepared (NOT run), docs rewritten.
- **Next (user)**: share Kaggle dataset `floorplan-stage2-final` (trained models, uploaded 2026-10-22, private) and the
  GitHub repo with the TAs; report, slides, videos (user writes the report).
- Old worktree `.claude/worktrees/stage2` (branch `stage2`, same history up to 5b37914) is no longer needed;
  removing it is the user's call.

## Novelty assessment (branch `novelty`, 2026-11-13 / 14; two rounds, final hybrid evaluated on held-out val, test NOT used)

Goal (user): large omissions on real CubiCasa5K drawings, end to end (no mask given), plus one hybrid for all sizes;
round 2 target (user, 2026-11-13 evening): at least 60-70% end to end on large omissions, same dataset only.
Write-up: `docs/novelty.md` (sections 1-5 round 1, section 8 round 2); code `src/novelty/` (README lists the files);
configs `configs/novelty/`; kernels `kaggle/kernels/novelty/`; results `outputs/novelty/`; choices `docs/decisions.md`
("Novelty assessment" and "round 2").
- **Final hybrid (round 2, v3c, "structure first")** = shipped pipeline + STRUCTURE NET (whole plan; inputs parsed
  map + wall-continuation rays + ink map + stroke map; marks missing walls; groups of them = region flags) +
  round-1 edit fill (v2) with walls drawn at wall probability >= 0.18 and the structure walls kept. Config
  `configs/novelty/pipeline_structure_c.yaml` (the default of `src/novelty/pipeline.run(image)`; other versions via
  env `NOVELTY_CONFIG`; v3b = the same with the first structure net, no stroke map). Checkpoints (git-ignored):
  `outputs/kaggle/novelty_structure2/` (struct4_s0, final), `novelty_fill_edit/`; earlier: `novelty_structure/`
  (v3b), `novelty_locate/`, `novelty_fill_ffc/`, `novelty_fill_plain/`; ablations `novelty_structure_abl/`; sets
  `novelty_data/`, `novelty_data2/`.
- **Held-out result** (151 drawings, `outputs/novelty/e2e_heldout_round2.md`), large omissions, nothing given:
  recovered (2 px) 11% (shipped) -> 52% (round 1) -> 75% (v3b) -> **80%** [73, 86]; strict fixed 13% -> 36% -> 48%
  -> **55%** [47, 63] (a perfect answer scores 72% strict); medium + large recovered 14% -> 51% -> 73% -> 80%, fixed
  14% -> 37% -> 51% -> 56%; small 42% -> 42% -> 42% -> 41%. Region flag on 3% of intact drawings (v3b 7%, round 1
  13%, shipped 1%). The user's 60-70% target is met by the 2 px rule, not by the strict rule (55%).
- **Caveat to carry into any write-up**: the gain needs the ink map (without it 55% recovered on tuning). Checked:
  no traces of erased walls in it; it shows the BLANK AREA a large omission leaves (all strokes cleared by the
  generators), so the numbers are for "a part of the plan not drawn at all".
- Protocol: everything chosen on the 212 TUNING val drawings, reported on the 151 HELD-OUT ones; each hybrid version
  (v1, v2, v3b, v3c) frozen first and run once on held-out. Kaggle quota counts session hours (30 / week).
- **Open (user's call)**: one test-split run of the final hybrid
  (`python -m src.novelty.eval_e2e --split test --test-approved-by-user`; DONE for v4c, round 4); merging `novelty` into
  `final` / `main`; a hybrid tab in the Gradio app (not built). Not attempted: the room-graph idea (report 8.2).
- Kaggle note: `kaggle/kernels/novelty/eval` is re-pointed per run with `kaggle/kernels/novelty/point_eval.py`.
- **Round 3 (2026-11-14, validity; user: "don't just chase accuracy; validity must be maintained")**: rule-based
  clean-up of region fills, `src/novelty/regularise.py`, config `configs/novelty/pipeline_valid.yaml` (v3d = v3c +
  clean-up; NOT the default, NOT frozen, NOT run on held-out). Tuning, 208 drawings, large: recovered 74% -> 73%,
  strict fixed 53% -> 46%, thickness spread / loose pieces / specks / unrelated additions clearly down, but the
  project's rule check not better (violations 24.3 -> 28.0, valid 12% -> 6%; truth 11.2). Write-up `docs/novelty.md`
  section 9, table `outputs/novelty/validity_tuning.md`. Open (user's call): make v3d the final hybrid and give it
  one held-out run; a fill model selected with validity (not attempted). Nothing of round 3 is committed yet.
- **Round 4 (2026-11-20, review feedback; user: address every review point, "architectural validity ... even at
  the cost of accuracy")**: the project's "valid" cannot judge fills on real drawings (the ORACLE, true layout pasted
  into the parser's map, has MORE rule violations than the fills). New: `src/novelty/soundness.py` (the review's
  three checks: walls connect, openings in walls, rooms closed + reachable; no ground truth), `repair.py` (rule
  repair after the clean-up, kept only if problems go down), selection of diffusion samples by soundness, configs
  `pipeline_sound.yaml` (v4) / `pipeline_sound_select.yaml` (v4b), ablation configs `pipeline_abl_*.yaml`,
  `completion_by_size.py` (the 46% by size + definition), `assessment_figures.py`, `soundness_report.py`.
  Selection rule written in decisions.md BEFORE the tuning results (kernels v14, v15); v4b failed its recovered
  guard, so v4c (v4b without the clean-up's wall-end rule, `pipeline_sound_select_c.yaml`) was added before its
  result and won. **Round 4 final = v4c, now the default of `pipeline.run`** (v3c = accuracy-first alternative).
  Held-out (kernel v16, each version once): large omissions sound 24% (v3c) -> **81%** [74, 86] (true layout 70%),
  recovered 80% -> 74%, strict fixed 55% -> 44%; medium sound 42% -> 83%; small and intact unchanged. Tables
  `outputs/novelty/soundness_{tuning,heldout}.md`, figures `outputs/novelty/assessment/`, write-up `docs/novelty.md`
  section 10, point-by-point `docs/review_response.md` (all coding points DONE). "The eight constraints" of the
  review = the parser v2 fine-tune losses (decisions.md, round 4); the reason given is my reading of the code: the
  user confirmed it (2026-11-21; plus: rules that added nothing were dropped).
  **Test split: ONE run of v4c done (user-approved, 2026-11-21, kernel v17; never run again)**: large omissions sound
  78% [74, 82] (true layout 68%), recovered 66% [61, 71], fixed 47% [42, 53]; medium sound 85%, recovered 63%;
  small fixed 40%; intact flags 6.7%. `outputs/novelty/soundness_test.md`, `docs/novelty.md` 10.6.
  **Test comparison (user request, kernel v18, once each, same cases)**: large recovered / fixed / sound: shipped
  8 / 10 / 36%, v3c 78 / 56 / 19%, v4b 68 / 49 / 85%, v4c 66 / 47 / 78%. v4b more sound than v4c on test at the same
  accuracy; v4c stays final (chosen on tuning). Rows `outputs/novelty/e2e_test_all_arms.csv`; poster figures
  `outputs/novelty/poster/`, guide `docs/poster_guide.md`. App: novelty tab (v4c / v4b / v3c, held-out gallery).
  The test split is now closed for good.

## Standing rules (always apply)

- **Never delete anything without asking** (files, folders, branches, worktrees, data). List and compare first;
  moving to a backup folder is the agreed fallback, only with approval.
- **Ask before every commit, push, Kaggle upload and Kaggle launch** (exception granted 2026-10-21: commit + push the
  final reorganisation to GitHub).
- **Never print, copy or commit the Kaggle token** (`~/.kaggle/access_token`, anything in `.kaggle/`); don't run
  `kaggle config view`.
- **UTF-8 only**: every text file; always `encoding="utf-8"` in Python.
- **Frozen settings**: Stage 1 `configs/stage1/app.yaml`; Stage 2 `configs/stage2/pipeline.yaml` +
  `detector_threshold.yaml`. Never tune them on what the app shows or on test data.
- **Choose on val, report on test.** Stage 1 test: done once (2026-10-20). Stage 2 test: run once (2026-10-22) plus
  one user-requested second evaluation (component-level omissions, all test drawings, same frozen settings);
  nothing is ever tuned on test results; smoke tests use the VAL zips.
- **Every self-made choice goes in `docs/decisions.md`** with its reason. This file keeps rules + status only.
- `python -m pytest tests/` must keep passing. Datasets stay off git; CubiCasa5K drawings are not redistributed.
- **Report / slides are the user's own writing** (IE643 plagiarism and AI-content rules): provide material, figures,
  numbers and checks, not report prose.

## How to work with me

I'm a mechanical engineering student, new to ML tooling. Explain simply; for every new file say what it does and how
it fits in. Small modular files, well commented, settings in `configs/`. Before big or hard-to-undo steps, explain the
plan first. Laptop RAM is limited: load models lazily, test the app by calling its handlers, few browser screenshots.

## Compute and environment

- Laptop: Windows, 16 GB RAM, **no GPU** → data prep, evaluation, the app. Conda env `floorplan` (Python 3.11):
  `C:/Users/lenovo/miniconda3/envs/floorplan/python.exe`; set `PYTHONUTF8=1`.
- Kaggle (T4 × 2, 30 GPU h / week, account `venkatesh007c`): training and the Stage 2 test. Code goes up as dataset
  `floorplan-code` (`python -m src.kaggle_prep code`, then `kaggle datasets version -p . -r zip` from inside
  `data/kaggle_upload/floorplan-code`). Kernels attach earlier kernels' outputs via `kernel_sources`.
  `kaggle quota` shows the GPU hours left. Zero budget.

## Layout

`src/stage1/` (Stage 1 code), `src/stage2/{data,constraints,parse,detect,complete,pipeline,interface}/`,
`src/{utils,distributed,kaggle_prep,kaggle_kernel,doc_lines}.py` (shared); `configs/stage{1,2}/`;
`tests/stage{1,2}/`; `outputs/stage{1,2}/` (committed results; `outputs/kaggle/`, `outputs/runs/` = local
checkpoints, git-ignored); `kaggle/kernels/stage{1,2}/`, `kaggle/notebooks/`; `demo_app/`; `docs/`. Folder READMEs explain
each part.

## Stage summaries

- **Stage 1**: 128 × 128 ResPlan maps, 6 classes, mask given. Final: size-adaptive hybrid (matched U-Net sketch →
  noised to t₀ = 400/600/700 → constraint fine-tuned diffusion, stochastic DDIM). Test: tolerant mIoU 0.846, validity
  88.0%, connectivity error 0.011 (U-Net 0.850 / 81.0% / 0.085). Lesson: constraint weights must be calibrated by
  gradient size. `docs/stage1_summary.md`.
- **Stage 2**: real CubiCasa5K drawings, 16 classes; PARSE (parser v2B) → CHECK (detectors v2 + v3 + rules C8, R5,
  ink checks, R6, R8 mix) → FILL (16-class hybrid; B4 for large holes) → VERIFY (best-of-N). Val numbers and weak
  spots: `docs/results.md`.

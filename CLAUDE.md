# CLAUDE.md

Context for Claude Code. Read at the start of every session, then `docs/decisions.md`, `docs/kaggle_workflow.md`, `docs/constraints_spec.md`.

## Project

IIT Bombay IE643 course project, solo. **Diffusion-based reconstruction of missing floor plan components.** Submission deadline **Sun 25 Oct 23:59**; everything must be finished **Tue 20 Oct night** (user away Wed 21 – Sat 24; Sun 25 = emergency only). Favour simple, working solutions.

## Status (updated 2026-10-18)

- **Stage 1 done**, tagged `stage1-final`. Stage 2 on branch `stage2` (worktree `.claude/worktrees/stage2`).
- Done: data v3; constraint library (G1-G4; K1 trains 14 constraints, G3 variant C); parser v1 (val mIoU 0.699, no fallback); K1 constrained vs control (24k steps each).
- Running: K2a (GPU0 16-class U-Net, GPU1 missing-component detector), launched 2026-10-18 09:06.
- Next: rules on full plans + omission benchmark (laptop); parser v2; guidance + verifier; pipeline; Gradio.
- **Stage 1 test evaluation: once, Tuesday morning, as the last GPU job** (postponed by the user, 2026-10-18).

## Standing rules (always apply)

- **Never delete anything without asking** (files, folders, branches, worktrees, data). List and compare first; moving to a backup folder is the agreed fallback, only with approval.
- **Ask before every commit, push, Kaggle upload and Kaggle launch.**
- **Never print, copy or commit the Kaggle token** (`~/.kaggle/access_token`, anything in `.kaggle/`); don't run `kaggle config view`.
- **UTF-8 only**: every text file; always `encoding="utf-8"` in Python.
- **Stage 1 app settings are frozen** (`configs/app.yaml`). Never tune on what the app shows.
- **Choose on val, report on test.** Stage 1 test set: evaluated once, Tuesday morning (last GPU job), frozen Day 7 configs.
- **Every self-made choice goes in `docs/decisions.md`** with its reason. This file keeps rules + status only.
- **Stage 1 files stay untouched until the Tuesday tidy.** New code only in the Stage 2 folders below. `python -m pytest tests/` must keep passing.
- Datasets stay off git (`data/` ignored); CubiCasa5K stays on Kaggle (only small reports come to the laptop).

## How to work with me

I'm a mechanical engineering student, new to ML tooling. Explain simply; for every new file say what it does and how it fits in. Small modular files, well commented, settings in `configs/`. Before big or hard-to-undo steps, explain the plan first.

## Compute and environment

- Laptop: Windows, 16 GB RAM, **no GPU** → data prep, debugging, evaluation, Gradio. Save RAM.
- Training on **Kaggle** (T4 ×2, ~30 GPU h/week; plan ~26 h this week). Workflow: `docs/kaggle_workflow.md`. Code goes up as dataset `floorplan-code` (`python -m src.kaggle_prep code`, then `kaggle datasets version -p .` from inside the folder). Kernels attach earlier kernels' outputs via `kernel_sources`. Training must checkpoint and auto-resume. Plan long 2-GPU runs at ~66 samples/s (T4 throttling).
- Conda env `floorplan` (Python 3.11), `pip install -r requirements.txt`; don't reinstall torch on Kaggle. Use `pathlib`. Zero budget.

## Stage 1 (done): mask-given inpainting on ResPlan

128×128 ResPlan label maps, 6 classes (wall, door, window, room, outdoor, background); the model gets the plan + a mask and redraws the masked area. **Final method: size-adaptive hybrid** — matched U-Net sketch → noised to t₀ = 400/600/700 (small/medium/large holes) → constraint fine-tuned diffusion, stochastic DDIM 50. Val: tolerant mIoU 0.842 (U-Net 0.846), validity 88.0% (80.4%), connectivity error 0.011 (0.092). Gradio app `python app/app.py`. **Key lesson: constraint weights set by loss value were ~100× too strong; calibrate by gradient size.** This becomes the report's "before prep review" chapter.
**Details, numbers, file map and commands: `docs/stage1_summary.md`.**

## Stage 2 (TA's clarified scope)

**Input:** a real black-and-white floor plan drawing (scanned/exported, any resolution and line width, no class info). The system identifies components, finds what is missing, and generates it with diffusion + explicit structural constraints. TA stressed component-level detection ("every room must have a door") and 10–15 component types.

**16 classes** (`configs/stage2/classes16.yaml`, `src/stage2/data/classes16.py`): wall, door, window, railing | living, bedroom, kitchen, bathroom, hallway, other_room | outdoor, background | toilet, sink, bath_shower, appliance.

**Pipeline: PARSE → CHECK → FILL → VERIFY**
1. **PARSE**: our own segmentation U-Net (ImageNet ResNet-34 encoder) trained on CubiCasa5K drawings → 16-class map. Greyscale, black-and-white augmentations (binarisation, line thickening/thinning, blur, noise, JPEG, scale 0.6–1.6×, 90° rotations, flips). v1 plain (CE + Dice), v2 fine-tuned with the local constraint losses. Inference at several scales; keep the scale where walls are ~8 px; rescale the label map so walls = 3 px.
2. **CHECK**: missing components = union of (a) rules: constraint violation maps thresholded on the full parsed plan, and (b) a learned detector (parsed plan + violation maps → per-pixel missing type), trained on simulated drafting omissions (+20% intact plans, + parser-like noise).
3. **FILL**: 16-class Palette-style x0-prediction diffusion on 128×128 windows around each hole, warm-started from Stage 1 checkpoints with class expansion; all constraint losses from step 0. Hybrid sampling as in Stage 1 (16-class U-Net sketch → re-noise to t₀ → DDIM).
4. **VERIFY**: constraint guidance in x0-space during DDIM + best-of-N (N = 8 eval, 4 demo) by fewest violations. Output: additions in red on the user's drawing (door = gap + leaf + arc) + a list ("Bedroom 2 had no door → door added on the hallway wall"). Manual mode: user draws a box, the model fills it.

**Constraints**: `docs/constraints_spec.md` is the spec (C1–C18 losses + R1–R4 check-only rules). One function per constraint → per-pixel violation map, used as training loss, sampling guidance, detection and verification. Weights calibrated by **gradient size**. Gates G1–G4 before training. Paths in the spec map to the Stage 2 layout (`src/stage2/constraints/ops.py`, `configs/stage2/constraint_constants.yaml`, `outputs/stage2/`).

**Data**: CubiCasa5K from Kaggle mirror `qmarva/cubicasa5k` (5,000 plans, official split), rasterised from SVG twice (never downsample label maps): parser set (greyscale + labels, median wall 8 px, long side ≤ 2048) and completion set (labels, wall 3 px). Kernel `kaggle/kernels/stage2_data`; details `docs/decisions.md` (Stage 2 data).

**Decisions (user, 2026-10-17; reasons in decisions.md)**: own parser (ImageNet encoder init OK); parser constraints in two steps (v1 plain, v2 fine-tune); CubiCasa's pretrained model only as fallback if by Sun noon our val wall IoU < 0.5 or door/window component F1 < 0.3; constrained vs control diffusion is the only ablation with its own run (others are evaluation switches); combined rules + learned detector; Stage 1 test set evaluated once on Sunday with frozen Day 7 configs (the new scope replaces the pending TA approval).

**Schedule**
- **Sat 17**: data + constraint library + gates; parser v1 kernel ~5–8 pm if data ready; K1 overnight (GPU0 constrained diffusion, GPU1 control).
- **Sun 18**: K2 (GPU0: 16-class U-Net with constraints, then Stage 1 test eval; GPU1: parser v2, then detector); rules on full plans; omission simulator; guidance + verifier; image-level erasure benchmark.
- **Mon 19**: join the pipeline; Gradio with drawing upload + manual mode; K3 = all evaluation.
- **Tue 20**: figures, README, full repo tidy, tests on 3–5 drawings from outside CubiCasa; user writes the report and records videos.

## Layout

Stage 1: `src/`, `configs/`, `tests/`, `app/`, `outputs/` (see `docs/stage1_summary.md`). **Stage 2**: `src/stage2/{data,constraints,parse,detect,complete,pipeline}/`, `configs/stage2/`, `kaggle/kernels/stage2_*/`, `outputs/stage2/`, `tests/stage2/`. Docs in `docs/`.

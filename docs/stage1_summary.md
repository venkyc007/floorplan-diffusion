# Stage 1 summary: mask-given inpainting on ResPlan (Days 1-8)

Moved out of CLAUDE.md on 2026-10-17 when the project scope changed to Stage 2 (see CLAUDE.md). Text kept as it was, section by section; this is the reference for the report's "before prep review" chapter. The tag `stage1-final` marks the last Stage 1 commit (f1d26ba).

Key commands: `python -m src.stage1.data.build_dataset --reuse-split`, `python -m src.stage1.data.make_benchmark`, `python -m src.stage1.train --config configs/<cfg>.yaml --out <dir>`, `python -m src.stage1.evaluate --checkpoint <best.pt> --benchmark val`, `python -m src.stage1.full_eval`, `python -m src.stage1.arch_eval`, `python -m src.stage1.stats`, `python -m src.stage1.selection`, `python app/app.py`, `python -m pytest tests/`.

## Final Stage 1 status (as of 2026-10-17, before the scope change)

**Stage 1 is finished.** The test set was evaluated once on 2026-10-20 with the frozen Day 7 settings: `outputs/stage1/stage1_test.md` (size-adaptive hybrid: tolerant mIoU 0.846, validity 88.0%, connectivity error 0.011). The scope then moved to Stage 2 (the TA's clarified task, which replaced the pending TA approval). The status list below is the Day 8 record.

**Where things stand:**
- **Done:**
  - dataset + controlled corruption (Days 1–2);
  - U-Net, diffusion and constraint fine-tune models (Days 3–6);
  - architectural metrics + statistics (Day 6);
  - final selection on val (Day 7): the size-adaptive hybrid;
  - Gradio interface (Day 8).
- **Final method:** size-adaptive hybrid.
  - The matched U-Net drafts the hole; the draft is noised to t₀ and finished by the constraint fine-tune.
  - t₀ = 400 / 600 / 700 for small / medium / large holes.
  - Val: tolerant mIoU 0.842, validity 88.0%, connectivity error 0.011.
- **Figures for the TA meeting** (made 2026-10-15/16; remake with `python -m src.stage1.interface.make_ta_figures`):
  - `outputs/stage1/interface/hybrid_samples.png`: 2 small, 2 medium and 2 large val-gallery plans.
  - `outputs/stage1/interface/hybrid_samples_large.png`: the first 4 large-hole plans, no selection.
  - What they show: walls, doors and windows are put back well. On the 29.5% room hole (plan 10691), the hybrid draws a plausible but different layout (no window, no inner wall). Show this honestly.
- **Open items, in order:**
  1. **TA approval** of the final method. Wait for it.
  2. **Final test evaluation, run once.**
     - No script does this yet: `src/stage1/full_eval.py` and `src/stage1/selection.py` read only `val_benchmark.npz`.
     - `data/benchmarks/test_benchmark.npz` exists locally.
     - Needs a test option plus a Kaggle kernel. The Kaggle dataset `floorplan-data` holds only train/val, so the test benchmark must be uploaded first; ask before the upload and before the launch.
     - Evaluate the final method plus the comparison methods (matched U-Net, hybrid 600, refine-600, nearest fill), with the same metrics and statistics as on val.
  3. **Stage-wise report + final documentation.** No report file exists yet.
     - Robustness material: per size bin and type in `outputs/stage1/selection.md` and `outputs/stage1/full_eval.md`.
     - Ablation material: constraints vs control in `outputs/stage1/finetune_ablation.md`; post-processing in `outputs/stage1/arch_eval.md`.
     - Interface section: `docs/interface.md`.
- **Repository state:**
  - `main` = GitHub `origin/main`.
  - The branch `worktree-gradio-interface` (worktree in `.claude/worktrees/gradio-interface`) is fully merged. It may be removed, but only after asking.
  - Untracked in the main folder, on purpose:
    - the 8 old screenshots `outputs/stage1/interface/01_start.png` … `08_editor_drawn.png`, never to be committed;
    - the 2 TA figures, not committed until the user says so.
  - Pre-merge copies of the interface files are in `C:\Users\lenovo\floorplan-diffusion-backup\2026-10-15-before-day8-pull\`.

## Deliverables

1. **Dataset + controlled corruption** — floor plan dataset, plus code that removes components (walls, doors, windows, rooms) in a controlled way by *type*, *location*, and *size*.
2. **Diffusion model** — takes a floor plan with missing parts and reconstructs them.
3. **Structure-aware constraints** — wall connectivity, doors/windows lying on walls, axis alignment, room closure (as losses, guidance, and/or post-processing).
4. **Component-level evaluation** — per-class IoU, component detection precision/recall/F1, structural metrics (e.g. connectivity, closure, door/window-on-wall rate).
5. **Robustness + ablation study** — vary corruption type/size/amount; turn constraints and model parts on/off.

Also required: a **Gradio interface**, **trained models**, **documentation**, and a **stage-wise report**.

## Data

- **ResPlan** dataset (github.com/m-agour/ResPlan): ~17k unit-level vector floor plans. Raw files go in `data/raw/`, processed tensors in `data/processed/`.
- **Exclude ResPlan's 683 "augmented" plans** (ids ≥ 100,000, listed in `split.json["augmented"]`). They are rotated/flipped/scaled copies of other plans, and ResPlan's augmentation **forgot to transform `inner`, `land`, `garden`, `parking`, `neighbor`** (verified: untransformed in 239/239 detectable cases) — so their outdoor areas and real scale are wrong.
- **Rooms are clipped to `inner`** (the indoor outline, +1 px): ~86 original plans label open-air areas outside the walls as rooms; those pixels become `outdoor`. Normal plans are unaffected (≤0.14% of room pixels).
- **Remove near-duplicates**, then **make our own train/val/test split** so that no near-duplicate pair ends up in two different splits. Dedup (see `src/stage1/data/dedup.py`): Stage 1 finds candidates by structure cosine (32×32, all 8 rotations/flips, ≥ 0.75); Stage 2 confirms with **tolerant mean per-class IoU** (wall/door/window/room/outdoor; a pixel matches if the other plan has the class within 1 px) on full 128×128 maps. A plan is removed only if tolerant IoU ≥ 0.90 with a **kept** plan (direct, no chain removals). Plain IoU was rejected: 2-px walls traced 1 px apart halve the wall/door IoU, so near-identical re-posts scored ~0.8. Kept plans with structure cosine ≥ 0.85 go in the same split (similarity groups).
- Pipeline: `python -m src.stage1.data.build_dataset [--limit 100] [--reuse-split]` (settings in `configs/stage1/data.yaml`). `--reuse-split` re-converts but keeps the existing exclusion + duplicate removal + split — **use it for any re-conversion so the split never changes**. Outputs in `data/processed/`: `train/val/test.npz` (`masks` (N,6,128,128) uint8 one-hot, `plan_ids`), `all_labels.npz` (label maps of all 17k plans, for checks), `metadata.csv` (all 17k plans incl. left-out ones; `split` ∈ train/val/test/removed_duplicate/excluded_augmented, `duplicate_of`, `duplicate_iou`, `similarity_group`, scale columns), `similar_pairs.csv` (candidates with cosine + tolerant IoU). Sanity checks + figures: `python -m src.stage1.data.checks`.
- ResPlan facts: coordinates are NOT metres — each plan is normalized to ~256 units on its long side (only `area` is metric). ~775 geometries are invalid, so rasterize by drawing coordinates (cv2), don't use shapely boolean ops. Load the pickle only via `src.stage1.data.resplan.load_plans` (allow-listed unpickler). `raw.githubusercontent.com` is blocked on the user's network — use `git clone` from github.com.
- Kaggle: upload `train/val/test.npz` + `metadata.csv` from `data/processed/` as a private Kaggle dataset rather than re-running the pipeline there (`all_labels.npz` is only needed for local checks).
- `data/` is git-ignored; never commit dataset files.

## Representation

- Each plan is a **128×128 map with 6 channels**: `wall`, `door`, `window`, `room` (room interior), `outdoor`, `background`.
- Channel order is defined once in `src/stage1/data/classes.py` — always import it from there.
- Channels are mutually exclusive (one-hot: exactly one class per pixel). Paint order outdoor → room → wall → window → door, so doors/windows are drawn on top of walls.
- ResPlan mapping: door = door + front_door; room = enclosed indoor rooms only (living, bedroom, bathroom, kitchen, storage, stair); outdoor = balcony, garden, land (yards/setbacks/light wells), parking, pool (ResPlan has no terrace class); inner/neighbor → background.
- **Framing rule — normalized per plan on the building's structure**: the bounding box of walls + doors (incl. front door) + windows is scaled so its longer side is 120 px (4 px margin each side), aspect ratio kept, centred. Nothing else affects the frame; balconies, outside stairs, garden/land/parking that extend beyond the 128×128 image are cropped. (Before 2026-10-07 balconies/rooms counted too, so a protruding balcony or outside stair shrank the building.) See `docs/data_notes.md`. Walls/doors are only ~2 px thick; doors/windows median ~27–30 px area, <0.1% under 6 px.
- **Real scale is preserved in metadata.csv**: `area_m2` (real area from the listing), `px_per_unit`, `metres_per_pixel` (= sqrt(area_m2 / (inner+wall area)) / px_per_unit), `wall_thickness_m`, and `scale_ok` (False for the 67 plans whose implied wall thickness is outside 0.10–0.40 m, i.e. a bad `area` value).

## Task design (Day 2): mask-given inpainting

- The model gets the corrupted plan + a mask of the region to repair and redraws what is inside; everything outside the mask is known.
- **Model input = 7 channels**: the 6 one-hot channels with masked pixels all zero + the mask as channel 7. **Target = clean 6-channel one-hot.** `src.stage1.data.corruptions.to_model_input(corrupted, mask)` builds the input; in corrupted label maps masked pixels are 255 (`UNKNOWN`).
- **Mask rule (no leaking the answer)**: never the component's exact shape — its bounding box + random 2–6 px padding per side (independent), clipped to the image. Region type: the random rectangle itself.
- **Corruption types** (`src/stage1/data/corruptions.py`, settings `configs/stage1/corruption.yaml`): door, window (one component), wall (one straight segment 8–40 px between corners/junctions/openings), room (one room component ≤ 50% of the building + its walls/openings within 3 px), region (random rectangle, sides 8–64 px). Metadata: type, location (exterior if removed pixels touch outdoor/background), position (center/edge, cutoff 0.55 of building radius), size bin (mask % of building: < 2.7 / < 7.7 / larger), mask area, n_components.
- **Training data**: `src.stage1.data.dataset.CorruptionDataset` — fresh random corruption per request, random rotation/flip first, types 0.2 each, 20% multi-removals (2–3 parts). ~1,000+ samples/s.
- **Benchmarks** (fixed, seeded): `python -m src.stage1.data.make_benchmark` → `data/benchmarks/{test,val}_benchmark.npz/.csv` (one sample per plan × type; 7,774 test / 7,772 val). Use **val for model selection, test only for final numbers**.
- **Metrics** (`src/stage1/eval/metrics.py`, `evaluate_benchmark()`): prediction outside the mask is replaced by the known pixels before scoring. Per-class IoU in the mask (pooled) + per-sample mIoU (absent classes skipped) + pixel accuracy; **tolerant IoU** for wall/door/window (match within 1 px, mask pixels only) + tolerant mIoU, reported next to the strict ones; door/window/room component P/R/F1 (4-conn, Hungarian, IoU ≥ 0.5, TP/FP/FN summed); conn_err (|#boundary components| diff, 8-conn, whole plan); open_on_wall; room_leak (rooms in/next to the mask touching outdoor/background); room_count_err = |#enclosed regions| diff (4-connected non-boundary regions not touching the image edge; a hole in a wall merges rooms, so holes are penalised); alignment (share of boundary pixels in the mask on ≥ 5-px horizontal/vertical runs).
- Baselines (`src/stage1/eval/baselines.py`): do-nothing, nearest-fill, ground truth → `outputs/stage1/baseline_results.md`. Tests: `python -m pytest tests/`.
- Every self-made choice is logged in `docs/decisions.md` — add new ones there.

## Models and training (Day 3)

- `src/stage1/models/`: `DiffusionInpainter` (Palette-style: UNet2DModel on 13 ch = noisy one-hot target in [−1, 1] (6) + condition (7); predicts x0; cosine schedule, 1000 steps; DDIM sampling (50 steps) that overwrites outside-mask pixels with the known plan at the matching noise level; argmax at the end; weighted MSE, mask × 5) and `UNetInpainter` (same backbone, 7 ch in → 6 logits, weighted CE, mask × 5). Common API: `model.loss(cond, target, mask)`, `model.predict(cond)` → label maps. Backbone: channels 64/128/256/256, 2 layers/block, attention at 16×16 + middle → **25.3M params** (user decided to keep this size; see decisions.md).
- `python -m src.stage1.train --config configs/train_{diffusion,unet}.yaml --out <run dir> [--smoke]`: AdamW + warm-up + cosine, fp16 on GPU, EMA 0.999 (diffusion), grad clip 1.0, `last.pt` every 30 min, **auto-resume** from `<run dir>/checkpoints/last.pt`, graceful stop at `time_limit_hours` (8.5). Validation every 2,000 steps on 256 fixed val-benchmark cases → `val_log.csv`, `best.pt` (by tolerant mIoU), `samples/`. **Keep `max_steps` unchanged across sessions.** Training needs the val benchmark (`python -m src.stage1.data.make_benchmark`) and `data/processed/train.npz`.
- `python -m src.stage1.evaluate --checkpoint <best.pt> --benchmark val|test` → full metric table. Compare/choose models on **val**; test only for final numbers.
- Checks: `python -m src.stage1.overfit_check` (→ `outputs/stage1/overfit_check.png`), `--smoke` runs, `python -m pytest tests/` (64 tests as of Day 8; the whole project now has its suite in tests/stage1 and tests/stage2). CPU is far too slow for real training (~7 s per step at batch 2) — train on Kaggle GPUs.

## Kaggle (Day 4)

- Workflow (upload code versions, launch, status, download, resume): **`docs/kaggle_workflow.md`**. Account `venkatesh007c`; private datasets `floorplan-data` (train/val label maps) and `floorplan-code` (`python -m src.kaggle_prep code`, then `kaggle datasets version -p .` run **from inside the folder**). Kernels live in `kaggle/kernels/<name>/`.
- **Ask the user before every upload and every launch. Never print, copy or commit the Kaggle token** (`~/.kaggle/access_token`); don't run `kaggle config view`.
- Measured on a Kaggle T4 (2026-10-08, single GPU): batch 32 max (64 runs out of memory); U-Net 1.20 steps/s, diffusion 1.17 steps/s (~37 samples/s); validation over 256 cases takes 3 s for the U-Net and 130 s for diffusion (50 DDIM steps).
- **Multi-GPU (DDP)**: Kaggle gives T4 ×2. `torchrun --standalone --nproc_per_node=2 -m src.stage1.train ...` (the kernels do this automatically via `src/kaggle_kernel.py`); plain `python -m src.stage1.train` = one process. `batch_size` is per GPU. Only rank 0 writes files and validates; clock-based decisions are broadcast from rank 0 (`src/distributed.py`). Validation uses `val_ddim_steps` (20); `src/stage1/evaluate.py` uses `ddim_steps` (50). `torchrun` doesn't work on Windows: to test DDP locally, start the processes by hand with `WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT` and `USE_LIBUV=0`.
- U-Net baseline run: `max_steps` 8,000 (fixed), about 2 h, full-val tolerant mIoU 0.775.
- **Plan long 2-GPU runs at ~66 samples/s**, not the short timing runs' 75: the T4s throttle (≈705 of 1,590 MHz, ~77 °C) after ~2 min. `gpu_usage.csv` logs clocks/temperature/power.
- **Diffusion run 1** (30,490 × 64, 8.5 h): val tolerant mIoU 0.698 (DDIM 20) < U-Net 0.794. Diagnostic (`outputs/stage1/diffusion_run1_diagnostic.md`): denoiser excellent (≥ 0.99 up to t = 600); one-shot at t = T = U-Net (0.792); best sampler DDIM 50 eta 1 ≈ DDPM 250 (0.753); samplers are structurally cleaner (conn_err ≈ 0, speckles at GT level) but commit to one plausible layout, which pixel IoU penalises. `[final x_t]` = `[last x0]`; the known-region reset makes no difference.

## Day 5: sampling + structural constraints

- Default sampler: stochastic DDIM 50 (`ddim_eta: 1.0`). Best usable setting so far: **refine from the one-shot prediction at t = 600** (`src/stage1/sampling_experiment.py`; tolerant mIoU 0.794 ≈ U-Net, connectivity error 0.016 against 0.133, room leak 0.014 against 0.052). See `outputs/stage1/diffusion_run1_sampling.md`.
- Constraint losses: `src/stage1/models/constraints.py` (clDice, openings, closure, alignment; on the softmax of x0 / 0.2, mask grown by 3 px; region-normalised). Configured under `model.constraints` (switch `use`, NOT `on` — YAML reads `on` as true), noise window `t_min..t_max`. Tests: `tests/stage1/test_constraints.py`.
- Fine-tunes: `configs/stage1/finetune_constraints.yaml` / `finetune_control.yaml` (identical except `constraints.enabled`; `tests/stage1/test_finetune_configs.py`), started with `--init <run-1 best.pt>`; kernels `kaggle/kernels/finetune_{constraints,control}/`. 26,000 steps × 64, lr 2e-5. **Constraints used: clDice 0.01 + closure 0.03 (ground-truth-relative); openings and alignment off.**
- **Constraint weights must be tiny (~0.01):** weights matched to loss values were ~100× too strong (the sharp softmax and skeleton ops amplify gradients) and wrecked reconstruction in the overfit test. Always run the diffusion overfit check (no constraints vs constraints, same seed) before a constrained run.

- **Full val results** (`outputs/stage1/full_eval.md`, `python -m src.stage1.full_eval`, kernel `kaggle/kernels/stage1/full_eval/`): best = fine-tune + refine-from-one-shot t = 600, tolerant mIoU **0.801** vs U-Net 0.775; constraint fine-tune = control on overlap, slightly better structure (not yet significance-tested).

## Day 6: matched U-Net, architectural metrics, statistics

- Matched-sample U-Net (`configs/stage1/train_unet_matched.yaml`, 30,000 × 64): full-val tolerant mIoU **0.846** > best diffusion 0.801; diffusion keeps clearly better structure (connectivity, room leak, validity). Logs: `outputs/stage1/unet_matched/`.
- Architectural rules: `src/stage1/eval/arch_metrics.py` (validity = no more violations than the ground truth on rules 1–6; only the "bridges" part of opening fit counts; doors may face background). Post-processing: `src/stage1/postprocess.py` (N = 6).
- Scoring is local: Kaggle kernel `arch_eval` saves predictions → `python -m src.stage1.arch_eval --preds outputs/kaggle/arch_eval/preds --out outputs` → `outputs/stage1/arch_eval.md`, `refine_tradeoff.png`, `arch_violations.png`, per-case CSVs → `python -m src.stage1.stats` → `outputs/stage1/stats.md` (paired bootstrap).

## Day 7: final selection (no training)

- Hybrid = matched U-Net prediction (saved `unet_matched.npz`) as the sketch → noised to t₀ → constraint fine-tune, stochastic DDIM (`hybrid_refineT` in `src/stage1/full_eval.py`, `--sketch`; kernel `kaggle/kernels/stage1/selection/`). `python -m src.stage1.selection` → `outputs/stage1/selection.md`, `pareto{,_room,_large}.png`.
- **Recommended final configuration: size-adaptive hybrid, t₀ = 400 / 600 / 700 for small / medium / large holes** — val tolerant mIoU 0.842 (matched U-Net 0.846), validity 88.0% (80.4%), connectivity error 0.011 (0.092). Single-setting fallback: hybrid t₀ = 600. Test (run once, 2026-10-20): tolerant mIoU 0.846, validity 88.0%, connectivity error 0.011 (`outputs/stage1/stage1_test.md`).

## Day 8: Gradio interface

- `python app/app.py` (port 7860; gallery once via `python -m src.stage1.interface.make_gallery`). UI only in `app/app.py`; logic in `src/stage1/interface/`; settings (frozen Day 7 selection) in `configs/stage1/app.yaml`. Design + tests + timing: `docs/interface.md`.
- Checkpoints load lazily (first use, then kept; ≤ 2 models, ~0.75 GB RAM). CPU: hybrid 29–52 s per repair, U-Net 1.4 s (`outputs/stage1/interface/cpu_timing.md`).
- Functional test (2026-10-14): 37/37 checks by calling the handlers. The full suite `python -m pytest tests/` had 64 tests at the time.
- Test the app by calling its handlers in Python (`setup()` then e.g. `repair(...)`), not through the browser — the laptop is short on RAM. Browser checks: at most 2–3 screenshots, at the end.
- Committed screenshots: only `outputs/stage1/interface/01_plan_and_mask.png` and `02_repair_result.png`. The 8 older screenshots (`01_start.png` … `08_editor_drawn.png`) in the main folder were deliberately left out (user decision). They are untracked; don't commit them.
- Git workflow used for Day 8 (user-approved, 2026-10-15): work on branch `worktree-gradio-interface` → push → fast-forward `main` (no force push). Before the pull, the main folder's 15 uncommitted interface files were moved, not deleted, to `C:\Users\lenovo\floorplan-diffusion-backup\2026-10-15-before-day8-pull\`. 9 of them were identical to the commit and 6 were older drafts with no unique work.

## Structural rules (for constraints and metrics)

- **Boundary = wall + door + window together.** Connectivity constraints/metrics treat all three as one boundary; doors and windows fill gaps in walls (they don't overlap them).
- **Gaps in interior walls without a door are legitimate open passages** (open kitchens, archways). Connectivity logic must tolerate them, not flag them as broken walls.
- **Room-closure constraints apply only to `room`, never to `outdoor`** (balconies/yards are open-sided by nature).

## Project layout

```
src/        Python source code (the reusable logic: data, models, constraints, evaluation)
configs/    YAML config files (hyperparameters, paths, experiment settings)
kaggle/     Kaggle kernels (kaggle/kernels/stage1/) and the Stage 2 test notebook
app/        Gradio web interface
docs/       Documentation (data_notes.md = data pipeline + benchmarks; decisions.md = every self-made choice)
tests/      Unit tests (python -m pytest tests/)
data/       Datasets (git-ignored)
outputs/    Figures + result tables (top-level *.png/*.md are committed); runs/ checkpoints & logs are git-ignored
```


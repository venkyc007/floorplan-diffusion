# Decisions log

Choices made without asking (or decided by the user, marked), each with a one-line reason. Newest at the bottom of each section.

## Data (Day 1)

- **Augmented ResPlan plans excluded** — their `inner`/`land`/`garden`/`parking`/`neighbor` were never transformed (see `docs/data_notes.md`), and they are copies anyway.
- **Rooms clipped to `inner` (+1 px)** — some plans label open-air areas outside the walls as rooms; "room" must mean enclosed indoor space.
- **Parking and pool count as outdoor** — neither is an enclosed indoor room.
- **Tolerant IoU (1 px) for dedup** (user decision) — plain IoU halves the score of 2-px walls traced 1 px apart.
- **Frame = walls + doors + windows bounding box, 4 px margin** (user decision) — protruding balconies/stairs no longer shrink the building.

## Corruptions (Day 2)

- **Wall segments found by morphological opening** (horizontal and vertical runs of ≥ 6 wall px; pixels in both = corners/junctions, removed to split) — walls are axis-aligned and ~2 px thick, so this splits cleanly at corners, T-junctions and openings without a skeleton.
- **Doors/windows also end a wall segment** — they are not wall pixels, so a segment naturally stops at an opening.
- **Wall segment length = longer side of its bounding box, kept if 8–40 px** — matches the requested range; shorter pieces are stubs, longer ones are whole facades.
- **Room "with surrounding walls" = room component + wall/door/window pixels within 3 px** — walls are ~2 px thick, so 3 px catches the full surrounding boundary.
- **Rooms smaller than 20 px ignored** — slivers left by rasterization, not rooms.
- **Room components whose bounding box covers > 50% of the building are skipped** — rooms joined by open passages form one huge component (masks up to 155% of the building); removing that is "redraw the plan", not "one room".
- **Region = rectangle centred on a random building pixel, each side 8–64 px, clipped to the image; no extra padding** — a random rectangle carries no shape hint, so it needs no padding; 8–64 px spans door-sized to multi-room holes.
- **Mask padding: 2–6 px per side, independently, clipped to the image** — as specified; clipping keeps masks inside the 128×128 frame.
- **Location breakdown kept as is, although windows (99%) and rooms (97%) are almost always exterior** (user decision) — by nature: windows sit in outer walls and most rooms have an outer wall, so interior/exterior comparisons mostly reflect doors, walls and regions.
- **"Removed component" for location: the removed pixels (room: room + its boundary ring; region: building pixels in the rectangle)** — so a room on the facade counts as exterior because its outer wall is removed with it.
- **Exterior = removed pixels 4-adjacent to outdoor or background** — as specified; 4-adjacency matches the component connectivity.
- **Position: distance of the mask centroid from the building centroid ÷ building radius (largest centroid-to-building-pixel distance); edge if ≥ 0.55** — dividing by the radius makes it comparable across plan shapes; 0.55 is the median over 1,500 train plans × 5 types, so the two bins are balanced.
- **Building area = wall + door + window + room pixels** (outdoor excluded) — size should be relative to the apartment, not its yard.
- **Size bins: small < 2.7% ≤ medium < 7.7% ≤ large (mask area % of building area)** — the tertiles (2.69%, 7.71%) of single-corruption masks pooled over 1,500 train plans × 5 types.
- **Corrupted label map marks masked pixels as 255 ("unknown")** — one compact uint8 array carries both the known pixels and the mask.
- **`n_components` = number of parts removed (1, or 2–3 for multi)**; region counts as 1 — a rectangle is one removal operation.

## Training dataset (Day 2)

- **Multi-removal: types drawn independently with the same type probabilities (repeats allowed), masks united; overlapping masks allowed** — simplest unbiased reading of "mixed types".
- **If a requested type has nothing to remove, fall back to the other types in random order and count the skip** — training can't skip a sample; the counts stay visible in `dataset.skips`.
- **Rotation/flip applied to the label map before corrupting** — as specified; corruptions are then drawn on the rotated plan.
- **Labels kept in memory as uint8 label maps (204 MB), one-hot built per sample** — 6× less memory than one-hot arrays.
- **Per-process random generator seeded from `torch.initial_seed()`** — DataLoader workers differ, and `torch.manual_seed` makes runs repeatable.
- **One OpenCV thread per worker process** — otherwise OpenCV's threads oversubscribe the CPU when several workers run.
- **Returned tensors are float32 as specified** — transfer between worker processes is the speed limit on Windows (~1,200 samples/s with 4 workers); if it ever limits GPU training, returning uint8 and converting on the GPU would cut transfer 4×.

## Benchmarks (Day 2)

- **One sample per (plan, type); the random seed is derived from (2026, plan id, type index)** — each sample is reproducible on its own, independent of processing order.
- **Saved as label maps (`inputs` = corrupted map with 255 in the mask, `targets`, `masks`), not 7-channel float arrays** — 7,774 × 7 × 128 × 128 floats would be ~3.5 GB; `corruptions.to_model_input()` rebuilds the 7-channel input exactly.
- **Stored in `data/benchmarks/` (git-ignored)** — fully reproducible from the seed with `python -m src.data.make_benchmark`.

## Metrics (Day 2)

- **Prediction outside the mask is replaced by the known pixels before scoring** — the model is given them; it should be neither rewarded nor punished there.
- **Per-class IoU pooled over the benchmark (Σ intersection / Σ union); mIoU = per-sample mean over classes present in pred or GT, then averaged over samples** — pooled IoU is the standard segmentation number; per-sample mIoU follows the spec's "exclude absent classes" rule.
- **Tolerant IoU for wall, door, window (user decision), reported next to the strict IoU** — a pixel matches if the other map has the same class within 1 px (as in dedup); tolerant IoU = matches / (total − matches), pooled over the benchmark. Plain IoU halves for 2-px walls drawn 1 px off.
- **Tolerant IoU compares only pixels inside the mask, on both sides** — otherwise known walls just outside the mask "match" true walls at the mask edge, and a model that draws nothing scored 0.07 (a unit test caught this).
- **Tolerant mIoU = per-sample mean using tolerant IoU for wall/door/window and strict IoU for room/outdoor** — tolerance matters for thin classes; for large areas it would only blur the score.
- **Pixel accuracy counts all 6 classes inside the mask (incl. background)** — background inside the mask is a real answer.
- **Component matching uses each component's full extent (not only the part inside the mask)** — composited pixels outside the mask are identical in pred and GT, so this measures the whole door/room.
- **Boundary components counted with 8-connectivity; door/window/room components with 4-connectivity** — wall pixels touching only diagonally are still one structural line; openings and rooms follow the spec's 4-connectivity.
- **Room leak considers rooms overlapping or bordering the mask (mask grown 1 px)** — a missing wall in the mask opens the neighbouring rooms even if no room pixel is drawn inside it (a unit test caught this).
- **Room count error = |#enclosed regions pred − gt| (user decision; replaces room-pixel components)** — regions = 4-connected components of non-boundary pixels that do not touch the image edge (edge-touching ones are the outside world: yards, open balconies, background). A hole in a wall now merges the rooms on both sides, so leaving holes is penalised (do-nothing 1.99 vs nearest fill 1.70; with room-pixel components do-nothing "won", 0.29 vs 0.91). Errors above 1 are real: a padded door mask often also cuts nearby walls and opens 3+ rooms at once. Courtyards/light wells fully enclosed by walls count as regions in both pred and GT. Tiny wall-gap pockets are negligible (6 of 500 test plans have one < 10 px).
- **Openings-on-wall and room leak are pooled ratios (Σ hits / Σ components); conn_err and room_count_err are per-sample means** — rates should weight every component equally; errors are per plan.
- **Alignment and openings-on-wall kept as is** (user decision) — they barely separate nearest fill from ground truth (it copies straight lines), but are meant to catch blobby/diagonal model output.
- **Alignment score = share of predicted boundary pixels inside the mask that lie on a horizontal or vertical run of ≥ 5 boundary pixels (morphological opening with 1×5 and 5×1 lines)** — straight axis-aligned walls are fully covered, diagonal/staircase or blobby strokes are not; ground truth scores 1.00. Undefined (NaN) when the prediction draws no boundary in the mask.
- **Rates with nothing to count (e.g. no openings predicted) are NaN, not 0 or 1** — "no data" must not look like a perfect or a failing score.
- **Markdown tables written by a small helper instead of pandas' `to_markdown`** — avoids an extra dependency (tabulate).
- **pytest added to requirements** — needed to run `tests/`.
- **`outputs/*.md` committed like `outputs/*.png`** — `baseline_results.md` is a report artifact; the larger CSV stays ignored.

## Models (Day 3)

- **Architecture kept exactly as specified (channels 64/128/256/256, 2 layers per block, attention at 16×16 + middle block) at 25.3M parameters, not 30–40M** (user decision, 2026-10-08: keep 25.3M) — with a 128×128 input the 4 levels run at 128/64/32/16 px, so "attention at 16×16 and below" is only the last level; the spec's architecture and size disagree, and the architecture is the more concrete instruction. To reach the range: 3 layers per block → 33.3M, or last level 384 channels → 40.8M (attention also at 32×32 only adds 1.3M).
- **Self-attention head size 32 channels** — a common default; gives 8 heads at 256 channels.
- **Condition channels stay 0/1 (not scaled to [−1, 1]) while the noisy target is in [−1, 1]** — in the condition, "all zeros" means masked; scaling to [−1, 1] would turn masked pixels into "all −1", an ordinary-looking value.
- **Training timesteps drawn uniformly from 0–999** — standard DDPM training.
- **Predicted x0 clipped to [−1, 1] (`clip_sample`)** — the target lives in that range; clipping stops sampling from drifting.
- **DDIM with eta = 0 (deterministic given the start noise), `set_alpha_to_one=True`** — standard DDIM; the last step then lands exactly on the predicted clean plan.
- **Known-region overwrite: the start noise outside the mask is also replaced by the known plan noised to the first timestep; after each step, outside-mask pixels get the known plan noised to the NEXT timestep (clean after the last step)** — keeps the known region consistent with the current noise level throughout, as specified.
- **Diffusion loss = Σ(weight × squared error) / (Σ weight × 6 channels), weight 5 inside the mask, 1 outside** — a weighted mean, so the loss scale does not depend on mask size.
- **U-Net baseline reuses `UNet2DModel` with a constant timestep 0** — "same backbone" exactly; the time embedding is then a constant learned bias.
- **U-Net cross-entropy normalised the same way (Σ weight × CE / Σ weight)** — same reason as the diffusion loss.
- **GroupNorm group count configurable (`norm_num_groups`, default 32; 8 for smoke models)** — tiny test models have 16 channels, which 32 groups can't divide.

## Training (Day 3)

- **Training counted in steps, not epochs; batch 32** — corruptions are random per sample, so "epochs" carry no meaning; 32 fits a 16 GB Kaggle GPU at 128×128 with fp16 for a 25M U-Net.
- **Learning rate 1e-4 (diffusion) / 2e-4 (U-Net), AdamW weight decay 0.01, 1,000 warm-up steps, cosine decay over `max_steps` (200k diffusion / 100k U-Net)** — common values for these model sizes; the U-Net's single-pass objective tolerates a higher rate. `max_steps` sets the schedule length; the 8.5 h time limit usually stops a session earlier and the next session resumes.
- **`max_steps` must stay the same across resumed sessions** — the schedule is recomputed from it (changing it makes the learning rate jump, seen in the smoke test).
- **EMA (0.999 per step) for diffusion only, as specified; validation and `best.pt` use the EMA weights; buffers are copied, not averaged** — EMA weights usually sample better.
- **fp16 via autocast + GradScaler on GPU only; gradients unscaled before clipping (max norm 1.0)** — clipping must see the true gradient size. fp16 could not be tested here (no GPU).
- **Checkpoints written to a temporary file and then renamed** — a session killed mid-save can't leave a broken `last.pt`.
- **`last.pt` holds model, EMA, optimiser, scheduler, scaler, step, best score, cumulative training hours and torch/CUDA RNG states; resume is automatic when `OUT/checkpoints/last.pt` exists** — "run the same command again" is the whole resume procedure.
- **DataLoader order and worker RNG are not restored on resume** — corruptions are random per sample anyway, so continuing with fresh randomness changes nothing meaningful.
- **Time limit measured from process start (includes data loading)** — the safe side for Kaggle's session limit.
- **Validation: 256 val-benchmark cases chosen by a seeded random draw (seed 1234), every 2,000 steps, diffusion noise seeded (0) so successive validations are comparable; best checkpoint by tolerant mIoU** — tolerant mIoU is the headline metric least sensitive to 1-px wall offsets.
- **Sample grids show 8 fixed val cases (input | prediction composited with known pixels | ground truth)** — the same cases each time make progress visible.
- **Logs as CSV (`train_log.csv` every 100 steps, `val_log.csv`), hours cumulative over sessions** — readable without extra tools.
- **`src/evaluate.py` added (not requested)** — scores any checkpoint on the full val/test benchmark with all metrics; needed to compare models next.

## Checks (Day 3)

- **Overfit check: tiny models (0.39M), Adam lr 3e-3, no clipping, 400 steps; pass = final loss < 50% of initial** — a quick "can it learn at all" test; the U-Net's loss spiked once mid-run at this high rate and recovered, which the real training's warm-up and clipping avoid.
- **Smoke settings: tiny models, batch 4, 20 steps, validation every 10 steps on 16 cases with 5 DDIM steps, checkpoint after every step** — exercises every code path (validation, best/last checkpoints, sample grids, resume, time-limit stop) in about a minute on CPU.

## Kaggle (Day 4)

- **Data uploaded as compact label maps (uint8, compressed): train + val 3.2 MB, plus train/val metadata; no test data** — 6× smaller than one-hot; `load_labels` reads both formats. The dataset is licensed CC-BY-4.0, like ResPlan.
- **Code shipped as a separate private dataset (`code.zip`; Kaggle unpacks it into folders)** — code can change without re-uploading data; the kernels handle both the unpacked and the zipped form.
- **Kernels are scripts that copy code + data into `/tmp/code` and write only results to `/kaggle/working`** — keeps the kernel output to the run folder.
- **The val benchmark is rebuilt on Kaggle from its seed and checked against `configs/benchmark_checksums.json` (SHA-256); the kernel stops on mismatch** — guarantees Kaggle scores use the same val cases as local ones (matched on the T4 run).
- **Batch size 32 for both models** — measured on a T4 (15.6 GB): 32 fits (10.9–11.0 GB peak), 64 runs out of memory.
- **U-Net run: `max_steps` 8,000, fixed** — 1.196 steps/s measured, so about 1.86 h of training plus about 6 min overhead, which makes the requested ~2 h run.
- **U-Net logging: loss every 50 steps; validation every 2,000 steps or 25 min, whichever comes first (`val_minutes`)** — to catch loss spikes like the overfit test's; U-Net validation costs only about 3 s.
- **Only one of the two T4s is used** — the code is single-GPU; using both (DataParallel/DDP) could nearly double throughput, but is not implemented.
- **Speed test reuses `train.py`'s exact training step (`train_step`)** — timings then reflect real training, EMA included.
- **Windows CLI workaround: `kaggle datasets create/version` run from inside the folder with `-p .`** — the CLI breaks on nested relative paths on Windows.

## Tooling

- **Always pass `encoding="utf-8"` when Python reads/writes text files** — Windows defaults to cp1252; a write with "−" failed and emptied this file once (rebuilt from the session record).

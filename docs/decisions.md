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
- **Only one of the two T4s is used** — superseded below (multi-GPU added).
- **Speed test reuses `train.py`'s exact training step (`train_step`)** — timings then reflect real training, EMA included.
- **Windows CLI workaround: `kaggle datasets create/version` run from inside the folder with `-p .`** — the CLI breaks on nested relative paths on Windows.

## Multi-GPU (Day 4)

- **Both T4s used via PyTorch DistributedDataParallel, launched with `torchrun` (user decision: multi-GPU)** — built into PyTorch (no new dependency, unlike Accelerate) and the standard, most reliable route; without `torchrun` the same code runs as one process on one GPU or the CPU.
- **Loss computed inside a `LossModule.forward()` wrapper** — DDP only synchronises gradients for work done in `forward()`, and our models compute the loss in `.loss()`.
- **Only rank 0 writes files, logs, checkpoints and sample grids, and only rank 0 validates (the others wait at a barrier; process-group timeout 60 min)** — one writer avoids clobbered files; diffusion validation (about 1 min at 20 DDIM steps) is far below the timeout.
- **Clock-based decisions (validate, checkpoint, stop at the time limit) are made by rank 0 and broadcast every step** — otherwise processes could disagree and deadlock; step-based logging is the same on every rank anyway.
- **EMA kept on rank 0 only** — it is only needed for validation and checkpoints, which rank 0 does.
- **`batch_size` is per GPU (32); 2 GPUs give 64 per step; learning rate unchanged** — 32 is the most a T4 fits; keeping the learning rate is the conservative choice (2× batch with the same rate is stable for AdamW at these values).
- **`DistributedSampler` gives each process its own share of the plans; seeds are `seed + rank`** — otherwise both GPUs would draw the same random corruptions. DDP copies rank 0's initial weights to all processes.
- **On resume with several processes, each re-seeds with `seed + rank + step`; the saved RNG state is used only in single-process runs** — the checkpoint stores rank 0's RNG state only.
- **The logged loss is averaged over processes; `train_log.csv` also records samples/s (training time only) and the GPU count** — makes the speed-up directly visible.
- **The Kaggle kernels log `nvidia-smi` utilisation of every GPU every 30 s to `gpu_usage.csv`** — direct evidence that both GPUs are busy.
- **Validation during training uses `val_ddim_steps: 20`; final evaluation (`src/evaluate.py`) uses `ddim_steps: 50`** (user decision) — validation becomes about 2.5× cheaper.
- **Tested locally: single-process smoke + resume (identical numbers to before), and a real 2-process DDP run on CPU (gloo backend), including resume and a synchronised time-limit stop** — `torchrun`'s own launcher fails on Windows (rendezvous store), so the two processes were started by hand with the same environment variables `torchrun` sets; on Kaggle (Linux) `torchrun` is used.
- **The U-Net baseline was not re-run on 2 GPUs** (user decision) — the single-GPU run had finished and the 2-GPU timing run already proved the speed-up (2.0×, both GPUs at about 90% utilisation).
- **Validation step counting starts from the last validation (`step - last_val_step >= val_every`)** — in the U-Net run, a time-triggered validation at step 1,833 was followed by a step-triggered one at 2,000; for diffusion each duplicate would waste about 1 minute.
- **Kernel helpers moved into the code bundle (`src/kaggle_kernel.py`); each `run.py` only copies the code and imports them** — one copy of the setup logic instead of one per kernel.

## Results so far

- **U-Net baseline: 8,000 steps × 32 samples (256,000 samples) on 1 T4, 1.83 h; full-val tolerant mIoU 0.775 (strict 0.708) against nearest fill's 0.489 (0.437)** — it had not plateaued: validation tolerant mIoU was still rising at the end (0.782 → 0.794 over the last 2,000 steps, while the learning rate decayed to 0). It beats nearest fill on every metric except connectivity error (0.178 against 0.066): the sample grids show speckled stray wall pixels on large masks.
- **TODO (fairness): after the diffusion results, retrain the U-Net baseline with the same number of training samples as the diffusion run (31,500 × 64 ≈ 2.0 M, about 7.9× the first run), on 2 GPUs** (user decision) — otherwise diffusion would be compared against a much shorter-trained baseline.
- **U-Net run summaries committed in `outputs/unet_baseline/` (logs, evaluation tables, sample grids; no checkpoints)** — small report artefacts; the 400 MB of checkpoints stay in the git-ignored `outputs/kaggle/`.

- **Diffusion run 1: 30,490 steps × 64 (1.95 M samples, 7.6× the U-Net's 256,000) on 2 T4s, stopped by the 8.5 h limit 1,000 steps short of `max_steps` 31,500** — it ran at 66 samples/s instead of the 75 measured in the 300-step timing run. Validation (20 DDIM steps) tolerant mIoU 0.698, below the U-Net's 0.794 on the same 256 cases; mostly plateaued.
- **Planning speed for long runs: about 66 samples/s on 2 T4s, not the short timing runs' 75** — the T4s throttle to about 705 MHz (of 1,590) at about 77 °C within about 2 minutes of full load (`gpu_usage.csv` of the diagnostic).
- **GPU log extended with SM clock, max SM clock, temperature and power** (user request) — to see throttling; the throttle-reasons field was left out because its name differs between `nvidia-smi` versions.
- **Diagnostic of run 1 (`outputs/diffusion_run1_diagnostic.md`): the denoiser is excellent (tolerant mIoU ≥ 0.99 up to t = 600), its one-shot prediction at t = T equals the U-Net (0.792 against 0.793), and stochastic sampling (DDIM 50 eta 1, or DDPM 250) is the best sampler (0.753)** — the gap to the U-Net comes from samples committing to one plausible layout, which pixel IoU penalises against the per-pixel best guess; the samplers win on structure (connectivity error ≈ 0 against 0.133; speckles 0.04 against 0.24, the ground truth itself has 0.05).
- **Diagnostic interpretation of "one-shot with the known region filled in": a clean known plan outside the mask in the noisy target at t = T** — the literal reading of "noised to t = T" would be pure noise again. It scores badly (0.528): the model never saw clean values at t = T.
- **The diagnostic ran on the 256 fixed val cases only; the full-val 50-step evaluation was not run** (user decision) — it is deferred until the sampling setting is chosen.
- **Run-1 summaries committed in `outputs/diffusion_run1/` (logs, config, GPU logs, diagnostic table, last sample grid; no checkpoints)** — same reasoning as `outputs/unet_baseline/`.

## Sampling (Day 5)

- **Stochastic DDIM (eta 1) is the default sampler (`model.ddim_eta: 1.0`)** (user decision) — the best sampler in the run-1 diagnostic (0.753 against 0.728 for eta 0) and structurally the cleanest; still reproducible with a fixed seed.
- **"Refine from one-shot": the one-shot prediction is used as a SHARP plan (argmax as one-hot), noised to t0, then finished with the stochastic DDIM steps at or below t0 (of 50: 10/20/30 steps for t0 = 200/400/600)** — re-noising a committed plan lets the model rewrite only uncertain parts; using its blurry continuous values would keep the blur.
- **Medoid of 4 = the sample with the highest mean tolerant mIoU against the other 3 (inside the mask); the pixel average of 4 = majority vote of their one-hot maps** — the medoid never looks at the ground truth; the vote is the simplest pixel average of discrete samples.
- **Result: refine from one-shot at t = 600 is the best usable setting** (tolerant mIoU 0.794 ≈ U-Net 0.793, best door/window F1, connectivity error 0.016 against 0.133, room leak 0.014 against 0.052, speckles at ground-truth level, 0.32 s per case). The oracle best-of-4 (0.845) is reported as an upper bound only. Not yet made the default: it should be confirmed on the full val benchmark.

## Structural constraints (Day 5)

- **Losses on the predicted x0, turned into probabilities by `softmax(x0 / 0.2)`, computed only in the mask grown by 3 px; one on/off switch (`use`) and one weight each** (user spec) — temperature 0.2 keeps confident pixels near one-hot (ratio e^10) while blurry ones stay soft.
- **Wall connectivity = 1 − soft clDice (Shit et al., CVPR 2021) between predicted and true boundary (wall + door + window); soft skeleton with 4 iterations** — walls are about 2 px thick, so 4 thinning steps reach the centre line.
- **Openings on walls = unsupported door/window probability, with "wall support" spread from wall pixels THROUGH touching door/window pixels (12 steps of 3×3 dilation)** — a plain "3×3 neighbourhood touches a wall" test would wrongly penalise the middle pixels of a real 2 × 10–20 px door; 12 steps cover doors up to about 24 px.
- **Room closure = room probability next to (3×3) outdoor/background probability** — as specified.
- **Alignment = Σ|gx·gy| / Σ(gx² + gy²)/2 of the boundary probability (Sobel gradients), inside the region** — horizontal/vertical edges have one gradient component near 0, so the score is about 0 for axis-aligned structure and about 1 for 45° strokes; normalising by edge energy makes it independent of how much boundary there is. Real corners give a floor (about 0.04 on the ground truth).
- **Openings and closure are normalised by the REGION size, not by the opening/room probability** — after the softmax every pixel keeps a tiny leftover probability of each class, so a region with no real door (or no room) divided a tiny unsupported amount by a tiny total and scored about 0.87 for nothing (found on a real val case).
- **The on/off key is named `use`, not `on`** — YAML reads the word `on` as the boolean true, which silently broke the configs (caught by the smoke test; now covered by a unit test).
- **Noise window `t_min <= t < t_max` instead of only `t < t_max`; the fine-tune uses ALL noise levels (t_min 0, t_max 1000), against the suggested "only t < 600"** — the t < 600 overfit variant also hurt at the first weights, and the passing settings were tested with all levels. Measured on run 1: below t ≈ 600 its x0 guess is already as clean as the ground truth (every constraint at the ground-truth level), so a t < 600 window would leave the losses almost nothing to correct; the errors appear at t ≥ 800 (clDice 0.0007 → 0.016, openings 0.002 → 0.006 from t = 800 to 999), where the x0 guess is still sharp enough to judge (tolerant mIoU 0.95 at t = 800, 0.79 at t = 999).
- **(Superseded by the overfit tests below.) First weights: clDice 1.0, openings 1.0, closure 3.0, alignment 0.01** — chosen from the run-1 magnitudes so each weighted term is about 25–55% of the base loss (about 0.0027) averaged over noise levels; alignment is kept small because run 1 already sits at the ground-truth floor, so a large weight would mostly erode real corners.
- **Constraint losses are averaged with a per-sample noise weight (0/1) without re-normalising by the number of samples inside the window** — simple; with the full window every sample counts.
- **Each loss part (mse and each constraint) is logged in `train_log.csv`, averaged over GPUs; validation now also logs conn_err, open_on_wall, room_leak, room_count_err and alignment** — so the effect of the constraints is visible during training, not only in tolerant mIoU. `best.pt` is still chosen by tolerant mIoU.
- **Compute overhead of the constraints: within measurement noise on CPU (8.8 against 9.1 s per step); planned as 3%** — pooling and convolution on 128×128 maps are cheap next to the U-Net.

## Constraint overfit tests and the fix (Day 5, evening)

Diffusion-only overfit on one fixed batch of 8 (tiny model, Adam 3e-3, 400 steps, seed 0); tolerant mIoU on that batch after 20-step sampling (eta 0 / eta 1):

| Variant | Final base MSE | Tolerant mIoU | Conn. error |
|---|---|---|---|
| No constraints | 0.0117 | 0.868 / 0.864 | 0.12 |
| All 4 at the first weights (1, 1, 3, 0.01) | 0.073 | 0.365 (eta 1) | — |
| All 4 at ¼ weight | 0.025 | 0.571 / 0.573 | 0.00 |
| All 4, only t < 600 | 0.028 | 0.376 / 0.405 | 0.25 |
| All 4, openings/closure/alignment ground-truth-relative | 0.074 | 0.213 / 0.231 | 0.12 |
| clDice + closure (weights 1, 3) | 0.051 | 0.327 / 0.346 | 0.12 / 0.25 |
| clDice only (weight 1) | 0.053 | 0.402 / 0.407 | 0.00 |
| clDice only, weight 0.01 | 0.0104 | 0.930 / 0.879 | 0.00 |
| **clDice 0.01 + closure 0.03 (chosen)** | **0.0099** | **0.947 / 0.968** | **0.00** |

- **Openings, closure and alignment are ground-truth-relative: `max(0, L(prediction) − L(ground truth))` in the same region** (user decision) — removes any reward for deleting real doors or shrinking real rooms; a perfect prediction costs 0. It did not, on its own, fix the overfit damage.
- **The damage came from the loss STRENGTH, not from which rule or from cheating** — clDice alone (ground-truth-based) hurt as much at weight 1, and matched/beat "no constraints" at weight 0.01. The first weights were chosen by matching loss VALUES to the base loss, but what counts is gradient size, which the sharp softmax (×5) and the skeleton operations amplify; about 100× too strong.
- **Fine-tune uses clDice (0.01) + ground-truth-relative closure (0.03); openings and alignment off** (user decision: fallback to the ground-truth-based clDice plus closure; weights scaled by 1/100 after the test above). The same settings are in the control config, with only `enabled` differing.

## Fine-tunes (Day 5)

- **Two fine-tunes from run 1's `best.pt` (EMA weights, loaded with the new `--init`; ignored when resuming from `last.pt`): `finetune_constraints` (constraints on) and `finetune_control` (off), identical otherwise (enforced by `tests/test_finetune_configs.py`)** (user decision) — a fair ablation: same start, data order, seed, schedule and steps.
- **Learning rate 2e-5 (user spec), 200 warm-up steps, cosine to 0 over 26,000 steps × 64 (2 T4s × 32)** — about 7.2 h of training at about 64 samples/s (the measured 66 with 3% constraint overhead), plus about 0.4 h of setup and validation: about 7.6 h, under the 8.5 h safety stop.
- **Both fine-tunes run at the same time, as two separate kernels** — they then finish before the 05:30 IST quota reset (about 02:30 if launched at 19:00) and use about 15.2 h of the 18.8 h left; running two GPU kernels at once worked earlier (the 2-GPU timing run ran alongside the U-Net run).
- **The control computes no constraint losses, so it runs slightly faster; it still stops at the same step** — equal training, not equal wall time.

## Fine-tune results (Day 5 night)

- **Constraint fine-tune vs control: no measurable difference on the 256 val cases (tolerant mIoU 0.7512 vs 0.7511 at 26,000 steps; structural metrics within noise)** — reported as a null ablation result in `outputs/finetune_ablation.md`; the full-val evaluation decides. Interpretation: at a strength that doesn't damage reconstruction, training-time constraints don't change the (already structurally clean) diffusion samples.
- **Fine-tune summaries committed in `outputs/finetune_runs/` (logs, configs, GPU logs, last sample grid; no checkpoints)** — same reasoning as `outputs/unet_baseline/`.

## Full val-benchmark evaluation (Day 6)

- **All 7,772 val cases, all metrics, broken down by type/location/position/size (`outputs/full_eval.md`, kernel `floorplan-full-eval`): fine-tune + refine-from-one-shot t = 600 is the best method — tolerant mIoU 0.801 (U-Net 0.775, run 1 refine 0.786, run 1 DDIM 50 0.740, nearest fill 0.489)** — it also has the best door/window/room F1 (0.693 / 0.684 / 0.953) and much cleaner structure than the U-Net (connectivity error 0.039 vs 0.178, room leak 0.026 vs 0.065, room count error 0.202 vs 0.301), and beats the U-Net on every type and size. The U-Net keeps a slight edge only on room/outdoor pixel IoU and is ~35× faster (0.010 vs 0.35 s per case).
- **Constraint fine-tune vs control on the full benchmark: identical overlap (0.801 both); structure slightly favours the constraints (room count error 0.202 vs 0.215, connectivity 0.039 vs 0.045; large holes 0.556 vs 0.592 and 0.112 vs 0.131)** — consistent but small; not yet tested for statistical significance (a paired bootstrap over cases would settle it).
- **Plain DDIM 50 gives the cleanest structure (connectivity error 0.005, room leak 0.005) but the lowest diffusion overlap** — it commits to one plausible layout; reported as the structure-first option.
- **Evaluation predictions are kept in the kernel's temporary storage; scoring runs on the CPU cores in parallel** — the predicted maps (~127 MB per setting) aren't needed afterwards, and CPU scoring outside the GPU processes keeps GPU time short.
- **The large-hole figure (`outputs/full_eval_large_holes.png`) re-samples 8 cases locally from the evaluated checkpoints** — the scored predictions weren't kept; the figure is labelled accordingly.

## Day 6: matched U-Net, architectural metrics, post-processing, statistics

**A. Matched-sample U-Net**
- **`configs/train_unet_matched.yaml` = the original U-Net config with only `max_steps` changed (30,000 × 64 on 2 T4s = 1.92 M samples, 98.4% of diffusion's 1.95 M); a test enforces "only max_steps differs"** (user decision: same architecture, data, augmentation, loss, mask weighting) — learning rate kept at 2e-4 although the batch per step doubled (as for the diffusion DDP runs).
- **Length from the original U-Net speed scaled to 2 GPUs with the observed long-run throttling: about 70 samples/s → 30,000 steps ≈ 7.6 h (10.6% margin under the 8.5 h stop; still under it at 66 samples/s)** — 30,490 steps (exact match) would leave only about 7% margin.
- **The kernel evaluates `best.pt` on the full val benchmark and saves its per-case predictions (`preds/unet_matched.npz`)** (user request) — so new metrics and statistics need no GPU.

**B.1 Architectural metrics (`src/eval/arch_metrics.py`)**
- **Every rule counts only items touching the mask grown by 3 px and is always reported next to the ground truth's own count on the same cases; a case is "valid" if, for rules 1–6, it has no more violations than the ground truth** (user spec).
- **Dead ends use the wall skeleton with spurs shorter than 3 px ignored, and an end is legitimate if a door, window, outdoor or background pixel lies within 3 px** — skeletonisation creates tiny spurs at corners; walls legitimately end at openings and at the outside.
- **Validated on 500 real ground-truth plans (val cases with their masks); rules that flagged many real plans were fixed:**
  - **Opening fit flagged 72.6% of real plans.** Cause: the wall-thickness probe right next to an opening's end crossed the perpendicular wall at corners/T-junctions. Fixed by probing 1–4 px beyond each end and using the narrowest cross-section, then by comparing thickness with the NARROWER end wall and requiring alignment only as "both end walls cover the opening's thickness span, at least one matching it". Real openings now pass bridges 98.1%, thickness 90.1%, alignment 81.6% — but still **27.8% of real plans fail the full opening-fit rule, so only its "bridges" check counts towards validity (2.8% of real plans flagged); thickness/alignment are reported as diagnostics.**
  - **Doors connecting two spaces flagged 8.8% of real plans: real front doors open onto background (the outside of the plot). Background now counts as an outside space → 1.6% flagged.** (Departs from the spec's "not wall/background"; only wall on a side is a failure.)
  - Final real-plan flag rates of the validity rules: dead ends 11.0% (kept: real open passages leave free-standing stubs, and validity compares with the ground truth per case), fragments 0.0%, opening bridges 2.8%, door spaces 1.6%, windows facing outside 0.6%, unreachable rooms 1.2%.
- **Sliver thresholds = 1st percentiles over 74,699 training room components: area < 72 px or inscribed width < 7.64 px** (`configs/arch_metrics.yaml`, `python -m src.eval.arch_metrics`).
- **Rectangularity = room area / bounding-box area, reported as a mean next to the ground truth's mean (0.863 on 500 real plans); the figure marks rooms below 0.6** — it describes shape, not a defect, so it isn't in the validity score.
- **Room reachability: rooms = enclosed regions; open gaps already merge regions; doors link the regions they touch (within 2 px); the outside (edge-touching areas) is one node; rooms outside the main group (the one with the outside, else the largest) are unreachable.**

**B.2 Post-processing (`src/postprocess.py`)**
- **Dead-end stubs inside the mask are first EXTENDED to a wall straight ahead within 4 px, otherwise REMOVED if shorter than N px; removed pixels take the most common non-wall class around them; pixels outside the mask never change** (user spec).
- **N is chosen by the highest architectural validity among the N that alter at most 1% of REAL plans** — applied to 500 real plans, the clean-up changed 0.2% (N = 4), 1.0% (N = 6) and 2.2% (N = 8) of them: larger N also deletes legitimate short stubs, so validity on predictions alone would favour the most damaging setting.

**B.3/B.4 Evaluation**
- **One Kaggle kernel (`floorplan-arch-eval`) only generates and SAVES per-case predictions for the GPU methods; all scoring (Day 2 + architectural metrics, post-processing, sweep plot, figures) runs locally on the CPU (`src/arch_eval.py`)** — iterating on metrics costs no GPU quota. Its predictions are new samples, so its numbers may differ slightly from `full_eval.md` (sampling noise).
- **The violation figure shows, for each violation type, a case where the prediction breaks the rule and the ground truth doesn't** (sources: U-Net, then the constraint fine-tune, then run 1).

**C. Statistics (`src/stats.py`)**
- **Paired bootstrap: 1,000 resamples of the val cases (seed 0); both methods scored on the same resampled cases with the reports' exact definitions (means for tolerant mIoU / connectivity / room count; pooled ratio for room leak; % valid for validity); 95% percentile intervals; "statistically clear" = interval excludes 0** (user spec).
- **"Best diffusion" = constraint fine-tune + refine-from-one-shot t = 600** — the best tolerant mIoU in the full evaluation (tied with the control at 0.801).

**Day 6 results (full val, 7,772 cases; `outputs/arch_eval.md`, `outputs/stats.md`)**
- **Matched U-Net (30,000 × 64, 7.06 h, all steps done): tolerant mIoU 0.846, clearly above the best diffusion (0.801) and the original U-Net (0.775).** The original U-Net comparison was unfair (8,000 × 32 = 0.26 M samples against 1.95 M): with equal samples the U-Net wins on pixel overlap and room count error, while diffusion keeps clearly better structure (connectivity error 0.039 vs 0.092, room leak 0.026 vs 0.039, validity 83.6% vs 80.4%; all paired-bootstrap clear).
- **Constraint fine-tune vs control: same tolerant mIoU and validity (no clear difference), but clearly lower connectivity error, room leak and room count error** (overall, room holes and large holes) — the first statistically supported effect of the structural constraints.
- **Post-processing: N = 6 chosen (alters 0.86% of real plans; N = 8 alters 1.85%, over the 1% limit).** It never changes tolerant mIoU by more than 0.001; it helps most where stubs are common (U-Net connectivity error 0.178 → 0.128, matched U-Net 0.092 → 0.070); for diffusion the gain is small (validity +0.3 points).
- **Refine sweep: t₀ = 200–600 give the same accuracy (0.801–0.803); t₀ = 800 trades 0.009 tolerant mIoU for 5× lower connectivity error (0.008) and +2.7 points validity; full DDIM 50 loses 0.04 more for little extra structure.** t₀ = 600 stays the main setting (it was chosen on the sampling experiment); t₀ = 800 is the "structure-first" option.
- **Remaining gap to real plans: unreachable rooms (0.08 per case for diffusion vs 0.012 in the ground truth) and dead ends (0.20 vs 0.13)** — room holes and large holes remain the hard cases (validity ~57–60% vs 95%+ for doors/windows/walls).
- **Violation figure: `violation_locations` now also draws unreachable rooms (the count is unchanged: `unreachable_rooms` = number of those masks), and every row uses a different case.**
- **Per-case results (`outputs/arch_eval/per_case/*.csv.gz`, 9 MB) are committed so the statistics can be re-run without predictions; the predictions themselves (`outputs/kaggle/*/preds/`, ~1.5 MB per method) stay local / in the Kaggle outputs.**

## Day 7: final selection experiment (no training)

- **Hybrid = the matched U-Net's SAVED full-val prediction (`unet_matched.npz`, the exact predictions already scored) → one-hot → noised to t₀ → stochastic DDIM 50 (eta 1, steps ≤ t₀) with the constraint fine-tune and the known-region reset** (user spec; `hybrid_refineT` in `src/full_eval.py`, `--sketch`). Re-using the saved predictions makes the sketch identical to the scored U-Net, so differences come only from the diffusion finish.
- **Diffusion curve filled in with refine-from-own-one-shot at t₀ = 700 and 900** (user spec). One Kaggle kernel (`floorplan-selection`) only saves predictions; scoring, plots and statistics run locally (`src/selection.py`), like Day 6.
- **Candidates: both U-Nets, all diffusion settings (run 1 + control at t₀ = 600; constraint fine-tune at t₀ = 200–900 and DDIM 50), all hybrids. Post-processed variants are left out** (post-processing changes tolerant mIoU by ≤ 0.001 and is an optional add-on to any setting).
- **Pareto front on (tolerant mIoU, validity), both maximised; a point is dominated only if another is ≥ on both and > on one (ties stay on the front). Knee = front point with the largest perpendicular distance to the chord between the front's end points, both axes min-max normalised over ALL plotted candidates; with fewer than 3 front points (no interior point) the front point with the largest normalised sum is used** (user spec for the main rule; the fallback is ours).
- **"Best hybrid" = the hybrid with the largest normalised tolerant mIoU + validity** (equal weight, same normalisation as the knee) — the user did not define "best"; this uses both selection axes like the knee does.
- **Size-adaptive option: the knee of each size bin's own front is used for that bin's cases; the combined per-case results are scored and bootstrapped like any setting.** The size bin is computable at inference (mask area relative to the building, both visible in the input).
- **All selection happens on val, so the chosen setting's val numbers are slightly optimistic; the test benchmark gives the final numbers.**
- **Results (`outputs/selection.md`, `pareto*.png`): the overall Pareto front is the matched U-Net + hybrids t₀ = 400–800; all 9 pure diffusion settings are dominated by a hybrid. Knees: hybrid t₀ = 400 overall (tolerant mIoU 0.844, validity 86.3%), t₀ = 700 for room holes and for large holes; per size bin: small 400, medium 600, large 700.**
- **Recommended final configuration: the size-adaptive hybrid (matched U-Net sketch → constraint fine-tune, t₀ = 400 / 600 / 700 for small / medium / large holes): tolerant mIoU 0.842, validity 88.0%, connectivity error 0.011, room leak 0.013, room count error 0.126** — against the matched U-Net: −0.004 tolerant mIoU (clear but tiny) for +7.6 validity points, 8× lower connectivity error, 3× lower room leak, lower room count error (all clear). Single-setting alternative: hybrid t₀ = 600 (0.842, 87.4%, 0.023).
- **Decision rule for size-adaptive vs the single knee: size-adaptive if it is clearly better (bootstrap) on validity AND connectivity error and loses < 0.005 tolerant mIoU (half a point). This tolerance is a judgement call set AFTER seeing the results** (it loses 0.002, so the choice is not sensitive to the exact value between 0.003 and any larger tolerance).

## Day 8: Gradio interface (`app/app.py`, design in `docs/interface.md`)

- **Settings frozen: the app uses exactly the Day 7 selection (size-adaptive hybrid t₀ 400/600/700, hybrid 600, matched U-Net, refine-600, nearest fill; DDIM 50 schedule, eta 1, known-region reset), read from `configs/app.yaml`; nothing was tuned on app output. The gallery holds only val cases** (user spec).
- **UI only in `app/app.py`; all logic in `src/interface/` (testable without a browser)** — `app/` cannot hold an importable `app` package next to `app/app.py` without a name clash.
- **Hole size at inference = mask area as % of the building measured on the U-Net-completed plan** (the clean plan is unknown in the app). On all 7,772 val cases it reproduces the benchmark size bin in 7,771 (99.99%), against 99.55% for "known building pixels + mask". Chosen on val before any app use; it changes no model or sampling setting.
- **Gallery: 24 val-benchmark cases, 8 per size bin, cycling through the corruption types that occur in that bin (a single door or window is never a large hole), seed 0; saved to `data/app/val_gallery.npz` and NOT committed** (dataset-derived, like all of `data/`); the app works without it (editor and uploads).
- **Upload pipeline = the data pipeline's framing** (structure bounding box → 120 px long side, centred, nearest neighbour, crop / background fill), after snapping every pixel to the nearest class colour. Our own choices: transparent → background; grey (within 12 RGB units) → unknown and always repaired; refuse if > 10% of pixels are > 60 RGB units from every class colour; 128 × 128 inputs whose structure already spans 112–124 px are kept unframed (re-sampling would distort 2-px walls); ≥ 100 wall pixels; sides 16–4,096 px; `.npy` loaded with `allow_pickle=False`.
- **Editor/mask read-back: the canvas is 4× (512 px); colours snap to the nearest class, then each 4 × 4 block takes its majority (removes blended brush edges); a pixel is masked if ≥ 25% of its block is brushed.**
- **"Suggest suspicious areas" = 4 rules (unreachable rooms, dead-end walls, floating fragments, doors not between two spaces) on the whole plan, a 4-px-padded box around each violation; labelled assistive / not exhaustive** (user spec; padding ours).
- **Pass/fail per rule: with a true plan (gallery) = no more violations than the true plan in the mask + 3 px (the evaluation's validity definition); without one = 0 violations**, with a note that real plans break some rules too.
- **Difference overlay: magenta = wall/door/window drawn by the repair; cyan = wall/door/window removed and amber = other class changes, both only where the input was known** — inside a gallery hole the input is unknown, so "removed/changed" would be meaningless there.
- **No speed-up that changes the computation**: on the laptop CPU one forward pass takes ~1.5 s for 4, 8, 10 or 16 threads, with or without `channels_last`; lower precision (bf16) would change the outputs of frozen settings, so the app runs in fp32 and shows per-step progress instead.
- **Checkpoints load lazily: a network is loaded on the first use of a method that needs it, then kept; start-up only checks that the files exist; at most the 2 networks are in memory** (user request, to save RAM on the 16 GB laptop). The load time (< 1 s each) is not counted in the per-repair runtime shown in the app.
- **The app was tested by calling its event handlers directly in Python, plus one short browser check with 2 screenshots** (user request, to save memory).
- **TA-meeting figures (`src/interface/make_ta_figures.py`) use the frozen app settings, seed 0, and val-gallery cases only, taken in file order (first 2 per size bin; first 4 large holes) — no cherry-picking** (user spec). A failure case is therefore shown as it comes: on the 29.5% room hole the hybrid leaves out a window and an inner wall.
- **Config override via the environment variable `FLOORPLAN_APP_CONFIG`** (used to test the missing-checkpoint message without touching real files).

## Tooling

- **Always pass `encoding="utf-8"` when Python reads/writes text files** — Windows defaults to cp1252; a write with "−" failed and emptied this file once (rebuilt from the session record).

# Stage 2 (from 2026-10-17)

## Scope and plan (user decisions, 2026-10-17)

- **Scope changed to the TA's clarified Stage 2** (real black-and-white drawings → find and generate missing components, 16 classes, PARSE → CHECK → FILL → VERIFY; see CLAUDE.md) — the TA's new scope replaces the pending Stage 1 approval. Stage 1 is tagged `stage1-final` and becomes the "before prep review" chapter.
- **Own parser** (U-Net, ImageNet-initialised ResNet-34 encoder) — ImageNet init is standard transfer learning, not a pretrained floor-plan model, so the parser stays our own work.
- **Parser constraints in two steps (v1 plain CE + Dice, v2 constraint fine-tune)** — a working plain parser exists early; the constraint effect on parsing is measurable as v1 vs v2.
- **CubiCasa's pretrained model only as a fallback** if by Sunday noon our parser's val wall IoU < 0.5 or door/window component F1 < 0.3 — protects the schedule without giving up our own parser by default.
- **Constrained vs control diffusion is the only ablation with its own training run**; all other ablations are evaluation switches (guidance on/off, verifier on/off, rules vs detector) — GPU budget (~26 of 30 h this week).
- **Missing-component detection = union of rules and a learned detector** — rules are explainable and need no training; the detector catches omissions rules can't express.
- **Stage 1 test set evaluated once, Sunday, with the frozen Day 7 configs.**

## Organisation (2026-10-17)

- **Stage 2 code in separate folders** (`src/stage2/`, `configs/stage2/`, `kaggle/kernels/stage2_*`, `outputs/stage2/`, `tests/stage2/`) — Stage 1 stays intact and runnable until the Tuesday tidy.
- **Stage 1 detail moved from CLAUDE.md to `docs/stage1_summary.md` verbatim** — CLAUDE.md is read every session and must stay short (< 8 KB); nothing is lost.
- **Work happens in a new worktree/branch `stage2`** (created from `f1d26ba`, = `main`) — the old `worktree-gradio-interface` worktree can then be removed (asked first).

## Stage 2 data (2026-10-17)

- **Source: Kaggle dataset `qmarva/cubicasa5k`** — checked complete: 5,000 plan folders, each with `model.svg`, `F1_original.png`, `F1_scaled.png`, plus the official `train.txt` (4,199) / `val.txt` (399) / `test.txt` (399); same layout as Zenodo 2613548, so no Zenodo download is needed. 3 folders are in no list and are not used.
- **SVG reading follows CubiCasa's own loader (`floortrans/loaders/house.py`)**: SVG coordinates are the pixel frame of `F1_scaled.png` (the SVG's width/height attributes are not); walls/railings/windows/doors by `id`, spaces by `class="Space <Type>"`, fixtures by `class="FixedFurniture <Name>"` with their transform (and the `FixedFurnitureSet` parent's). Checked by overlay on two sample plans before the full run.
- **Room grouping = CubiCasa's own 12-class grouping (`rooms_selected`), then the user's mapping** (Entry → hallway; Storage/Garage/Undefined → other_room). So Dining/Lounge/EatingArea → living; Sauna → bathroom; DraughtLobby/HallWay → hallway; Closet/DressingRoom *spaces* → other_room (via Storage); CarPort → other_room (via Garage); every other type (Room, Hall, Office, Utility, …) → other_room.
- **Fixtures**: toilet = Toilet, Urinal; sink = all sink types + WaterTap; bath_shower = Bathtub(Round), Shower, ShowerCab, ShowerPlatform, Jacuzzi; appliance = ElectricalAppliance (all sub-types), Wood/Gas/IntegratedStove, Dishwasher, GeneralAppliance, WashingMachine. Ignored (room/wall underneath kept): closets, coat racks, counter tops, housings, fireplaces, chimneys, sauna benches, cabinets, shower screens, SaunaStove, Misc. Departures from CubiCasa's grouping (flagged to the user): showers and Jacuzzi counted as bath/shower, WashingMachine as appliance, SaunaStove ignored.
- **Fixtures are painted last but never over wall/door/window/railing** — fixture outlines often overlap the wall they stand against by a pixel or two; painting them over the wall would cut walls. The share of fixture pixels affected is reported in `data_stats.md`.
- **Spaces painted outdoor first, then rooms** — a room drawn over an outdoor polygon keeps the room.
- **Wall thickness per plan = length-weighted median of the wall polygons' thickness**, computed exactly for rectangles from area and perimeter (t, L = roots of x² − (P/2)x + A); only elongated pieces (L ≥ 2t) count — square corner pieces have no meaningful thickness.
- **Scale: parser set s = 8 / t (long side capped at 2048 px, then walls < 8 px, recorded per plan); completion set s = 3 / t.** Labels drawn from the polygons at each scale with sub-pixel precision (cv2.fillPoly, 4 fractional bits; pixel-centre shift −0.5).
- **Parser image = the higher-resolution of F1_original / F1_scaled, greyscale (transparent → white), resized with area averaging (cubic when enlarging)** — F1_original is the same drawing uniformly rescaled (checked: identical ratio in x and y); used only when its x/y ratios agree within 2%.
- **Storage: PNGs inside uncompressed zip files per split** (`parser_{split}.zip`, `completion_{split}.zip`) — few output files, no pickles, readable without unpacking.
- **Constants from the train split only**, at completion scale; percentiles in `configs/stage2/data.yaml`. Square widths use the chessboard distance (largest odd square = 2·maxDT − 1), because opn() with a k×k square removes a component exactly when that width is < k. k_clear = 2·floor(p2 door-to-fixture distance) − 1 (so ~2% of real doors violate C14); k_fix = 2·ceil(p98 fixture-to-room distance) + 1. C6 adjacency counts pixel pairs touching directly vs separated only by boundary ≤ 6 px.

## Stage 2 constraint library (2026-10-17, `src/stage2/constraints/`)

- **One function per constraint returning a (B, 1, H, W) violation map; a registry holds each one's default mode (absolute / GT-hinged / GT-relative) as in the spec; `constraint_loss()` = masked mean over M = hole grown 3 px (7×7), always in fp32.** Without a hole (detection) M is the whole image.
- **Operators pad dilation with −inf and treat the outside as +inf for erosion** (spec), so rooms cut by the 128-px window border are not eroded from the border (unit-tested).
- **Even-sized openings dilate with the mirrored window of their erosion** — otherwise an even kernel (e.g. Lmin_door = 8) shifts the result by 1 px and flags the last pixel of every valid door (found by G1).
- **C1 map form = dead ends of straight wall runs, not soft-skeleton end points** (departure from the spec): the soft skeleton of 3-px walls breaks apart at T-junctions and corners and fired 7 times on the valid toy. A dead end = the last pixel of a horizontal-only (or vertical-only) wall run of ≥ T_wall + 2 px whose next pixel is not wall/door/window/railing/outside/beyond the image. The C1 training loss stays soft clDice against the ground truth (as in Stage 1).
- **C2 floods from known structure through all boundary pixels (wall + door + window + railing), not walls only** — a wall piece between two openings inside the hole touches only openings and would otherwise count as floating.
- **C3 = wall pixels on no horizontal or vertical run of ≥ T_wall + 2 px (line openings), not the Sobel product** (departure): Sobel x·y responses also fire at every wall end and corner, so a valid Manhattan plan is not ≈ 0. This is the Stage 1 alignment metric made differentiable.
- **C4 "too thin" uses an opening with T_wall (3 px)**, as in the spec's `opn(W, 3)`.
- **C5 continuation lines are traced from known wall runs ≥ T_wall + 2 px, through hole or wall pixels only, for 2·T_wall px into the hole** — a known wall that already ended at a door never "continues".
- **C8 training seed = rooms/outdoor on the window border, flooded through the real plan's walkable pixels** ("spaces on the window border" in the spec); inference seed = the largest 4-connected walkable component (computed on the CPU, no gradient).
- **C11 counts the outside background as a valid door side** (spec: rooms + outdoor only) — Stage 1 found that real front doors open onto background (8.8% of real plans flagged otherwise).
- **C12/C11 probe distance k = T_wall + 2.**
- **C16 support = walls, windows and railings** (a sink under a window stands against the wall too); flood fill 32 steps as in the spec.
- **C17 is GT-hinged per instance: relu(smax(required fixtures in Y) − smax(in P)) over each GT bathroom/kitchen instance (room + the fixtures standing in it) fully inside the window and overlapping the hole** — the spec's `1 − smax(P)` is ~0.3 even for a correct prediction because smax (temperature 0.1) over a room only partly covered by its toilet stays below 1.
- **C18 searches outdoor within one wall thickness (2·T_wall + 1 square, spec: 3)** — a 3-px railing's far side is 3 px from the outdoor pixels, so a 3×3 dilation flagged valid railings.
- **R1–R4 are discrete (numpy) on label maps; R1 room-area ranges and R3 forbidden door pairs need CubiCasa statistics that the first data kernel did not collect (placeholders: R1 checks nothing; R3 forbids bathroom–kitchen).**
- **G1 toys use toy-sized constants (`toys.TOY_CONST`)**; the real constants are for whole CubiCasa plans. Toy bathroom k = 5 because the bath and toilet leave a 6-px corridor — which shows that **k_c should be a per-pixel width percentile, not the per-component one the data kernel computed** (C7 flags thin PARTS of rooms). To fix before G2.

## Stage 2 data v2/v3 and parser v1 (2026-10-17, evening)

- **User decisions applied:** appliance = sub-types with >= 50% of instances in kitchens (survey pass, `configs/stage2/appliance_subtypes.yaml`); Hall -> hallway, CarPort -> outdoor; untyped plans (raw "Undefined" > 50% of room pixels) only in the parser set with room pixels = 255; C6 excludes pairs with contact rate >= 10%, never bathroom pairs, always outdoor-background; k_c = p2 over pixels of the largest surviving opening; k_clear = p10, C14 GT-hinged; R1 = p1/p99 room areas; R3 forbidden = pairs on < 0.5% of door connections, bathroom pairs decided by the user (bedroom-bathroom and bathroom-outdoor allowed, bathroom-background forbidden).
- **Room instances (R1, C17) include their fixtures: fixtures take the room type by growing rooms into fixture pixels step by step** — a straight-line nearest room can lie behind a 3-px wall and created fake 6-12 px rooms in testing.
- **Black padding -> white: near-black (<= 10) region touching the border, >= 0.5% of the image, >= 98% labelled background.** A first "rectangle-like" test missed plan 11784 (two padding blocks joined by a thin line); the parser applies the final rule when loading each drawing, so the data did not need a fourth rebuild. v3 counts (old rule): 40 plans with transparency, 16 with padding.
- **Parser crops with < 5% non-background pixels are re-drawn (max 10 tries)** (user).
- **Parser line thinning is undone if it erases > 40% of the ink** — thin-outline drawings vanished in the augmentation figure.
- **Parser lr: warm-up, then cosine over the session's time budget (2.75 h), not a step count** — the run is stopped by time; a step-based schedule would need a speed guess.
- **OOM safeguard: whole-drawing validation falls back to 1024-px overlapping tiles on CUDA OOM; `expandable_segments` allocator** (no earlier record of the approved safeguard was found; this is what was implemented).
- **Warm start (class expansion): the 16-class U-Net reproduces the Stage 1 U-Net exactly; the diffusion matches exactly up to t = 200, 96% at t = 400 and ~65% at t >= 600.** Expected and accepted (user): the 10 noisy room-type/fixture input channels carry independent noise, so the copied "room" weights cannot reproduce the single old channel at high noise; the conv_in bias absorbs the constant offset (exact at t = 0).

## Gates G2-G4 and K1 settings (2026-10-17, night)

- **G2 (1,000 real windows): C5, C9, C11, C12 switched to GT-hinged (14.6-23.9% of real windows violate); C4 (42.9%) and C7 (40.5%) kept, already GT-hinged** (user). `outputs/stage2/constraint_gt_violations.md`.
- **Calibration (GPU, 50 x 8 windows, t uniform 0-400; median over batches with signal; < 10 batches -> median weight; cap 10x the median weight)** (user rules). C2, C5, C10, C13 hit the cap. `outputs/stage2/constraint_weights.md`.
- **G3 first run failed (all 18, full rho: -6.1 accuracy / -3.8 mIoU). Re-run with variants (user): all four passed within max(2, OFF seed spread 1.3/1.4) points; by the user's order C, D, A, B the choice is C = the 14 uncapped constraints at full rho (-1.8 / -1.1).** C2, C5, C10, C13 are left out of TRAINING only; they stay on for guidance, detection and the verifier. `outputs/stage2/gate_g3_variants.md`.
- **G4 (T4, batch 32): 1.40 s/step with constraints vs 0.90 without (1.55x, K = 96); K = 48 gave 1.38 vs 0.97 (1.43x): K barely changes the constrained step, so K stays 96.**
- **K1 fairness: max_steps = 24,000 for BOTH runs** (user) — the constrained run at 1.40 s/step fits ~9.3 h + validation in the 10.5 h limit; the faster control stops at the same step, with the same cosine schedule over max_steps.
- **The warm-started diffusion matches Stage 1 only ~65% at t >= 600; expected (new room-type channels start undecided), not fixed** (user).

## Schedule change (user decision, 2026-10-18)

- **The Stage 1 test evaluation is POSTPONED to the end: run once, on Tuesday morning, as the last GPU job, with the frozen Day 7 configs.** (Replaces "Sunday" in the original plan.)

## Rule-based detection on full plans (2026-10-18)

- **Thresholds per rule on candidate score (summed violation pixels), tuned on 200 intact typed val plans; greedy: the rule with the most false alarms is tightened until the whole detector has <= 0.5 per plan** — ends at ~0.03 per rule. C1 (dead ends, 5.8 raw per real plan) is effectively switched off.
- **C8 and C17 use exact discrete versions on full plans** (walkable components with cv2; fixtures by room instance): the soft flood fill would need ~H+W steps on whole plans.
- **Omission size bins: < 0.5% / < 2% / larger of the building area** (own choice; Stage 1 tertiles were for ResPlan masks). Location exterior = omitted pixels touch outdoor/background.
- **Result: real plans break the rules often (about 57 raw candidates per intact plan), so at <= 0.5 false alarms the rules find only 7% of single omissions (rooms 17%, doors 10%, others <= 4%; `outputs/stage2/rules_check.md`).** Rules alone cannot carry CHECK; the learned detector must.

## Rules diagnostics and sampler (2026-10-18)

- **C8 walks from the main flat (most room pixels) and from every walkable area containing an exterior door** — multi-flat/multi-floor images made every room of the other unit "unreachable" (raw false alarms 1.96 -> 0.45 per intact plan). TA case: 100% of truly cut-off rooms found raw; the 15% "misses" stay reachable through open passages.
- **False-alarm budget re-split (user): C8, C17, C6, C12 0.1 per plan each, the rest share 0.1.** Recall on single omissions 7% -> 8%; false alarms 0.52 per plan on held-out intact plans.
- **Sampler (spec section 5): guidance on x0 in absolute mode over the 17 window-usable constraints (incl. the 4 left out of training) with the calibrated weights; only hole pixels change; x0 clipped to [-1, 1]. Verifier: blobs of V > 0.5 in M per constraint + R1 (rooms inside the window) + R3; C17 and R4 not usable on windows. Ties -> pixel agreement with the U-Net sketch.**

## Detection on whole plans (2026-10-18)

- **Detector proposals: blobs of (1 - p(none)) >= 0.9 with >= 40 px; type = missing class with the highest mean probability in the blob; box + 3 px** — the best mean recall (67%) at <= 0.5 false alarms per intact plan over a threshold x blob-size sweep on 300 whole val plans (chosen and reported on the same plans). `configs/stage2/detector_threshold.yaml`.
- **"Both" = detector + strong rules (C8, C17, C6, C12), merged**: same mean recall (67%), +3 points on doors and fixtures, but 0.85 false alarms per plan (rules add ~0.35). Rules alone: 7%.
- **Rule ceiling: 59% of single omissions are visible to some rule before thresholds, but only 11% of those survive the thresholds; windows (8%) and fixtures (12%) are almost invisible to rules.**

## CHECK final setting and pipeline (2026-10-18)

- **Final CHECK = detector (0.9 / 40 px) + C8** (user rule: + C8 if <= 0.6 false alarms per plan). Held-out val (159 plans not used for the setting; 59 intact, 100 x 2 omissions): 0.31 false alarms per intact plan, mean recall 62% (door 28%, window 73%, wall 83%, railing 52%, fixture 38%, room 97%). The detector already takes the rule violation maps as input, so it is itself the rules + learned combination.
- **Pipeline parse: parse once at scale 1, measure the predicted wall thickness (2 x median distance transform on the wall ridge), re-parse at the configured scale closest to 8 / t only if |t - 8| > 1.5 px** — the same choice as trying every scale, with 1-2 parser passes instead of 5.
- **Holes are cut to at most 100 px per side so they fit a 128 window; t0 by hole share of the building (Stage 1 bins 2.7% / 7.7%); eta = 0.3 placeholder; N = 4.**
- **CPU: ~3 min per hole (4 samples x up to 35 DDIM steps x guidance); parse 4-7 s, detect 2-8 s; 8-9 min per drawing with 3 holes.**

## Chunk 7: benchmark whitening, fast mode, t0 diagnostic (2026-10-18)

- **Benchmark whitening erases only the removed component's own label pixels (its rasterised SVG polygon), dilated 2 px (5 x 5 kernel), never a box** (user).
- **Per-type detector thresholds: tried and rejected** (user). Greedy per-type search (grid 0.5-0.995, start 0.995, doors weighted x2, railing/fixtures x0.5, budget 0.4 false alarms incl. C8) gave held-out mean recall 58% vs 61% for the single 0.9 and doors 17% vs 22%: the greedy stopped early with budget unused and left doors stricter. Kept: single 0.9 / 40 px + C8. Code stays in `detection_eval.py --per-type` for the record.
- **Fast mode (live app only): N = 2, 15 DDIM steps from t0, guidance on the last 5 steps; full mode (all evaluation numbers): N = 4, the steps <= t0 of a 50-step schedule, guidance at every t <= 600** (user). On 30 val holes (not ~50: CPU time, ~4 min per hole in full mode): pooled hole mIoU 38.1 full vs 39.1 fast, violations 2.80 vs 3.23, 194 s vs 40 s per hole. Fast misses the 30 s target (about 40 s). `outputs/stage2/fast_vs_full.md`.
- **t0 diagnostic (user): the hybrid beats the U-Net sketch alone at every t0 tried.** Same 30 holes, per-hole means: U-Net tolerant mIoU 67.1, validity 27%, 10.7 violations; hybrid fast at t0 200-600: 69.2-69.7, 73-77%, 3.3-3.8. The four t0 values are within one hole of each other. **Chosen t0 = 300 for the pipeline (fixed, no size bins)** by the user's rule (fewer violations / higher validity than the U-Net, tolerant mIoU within 3 points): highest validity (77%, tied with 400), then fewest violations (3.70 vs 3.77). Validity = no more thresholded verifier violations than the ground-truth window (Stage 1 rule on the 16-class verifier counts). Per-hole means differ from fast_vs_full.md's pooled numbers (rare classes weigh more when pooled). `outputs/stage2/t0_sweep.md`.
- **Full-mode row at t0 = 300 not finished** (stopped at 13/30 holes: laptop out of memory); the t0 choice rests on the fast rows.

## Overnight 2026-10-18/19 (user asleep; own choices logged here)

- **Realistic erasure (`src/stage2/data/erasure.py`), used everywhere (check set, examples, door evaluation): the component's label pixels grown 2 px are filled with cv2.inpaint (Telea, radius 3)** (user). Own additions after looking at the crops: (1) the walls / openings / fixtures within 16 px of the spot are masked during inpainting and restored after, so the fill comes from the room side only (without this, fixtures against a wall got a dark wall-coloured smear); (2) the 2 px growth never takes pixels of other walls / openings / fixtures (it nicked the adjacent wall). `outputs/stage2/erasure_check.png`. Known leftovers: a textured fill (tiles) becomes smooth in the spot; fixture ink drawn outside its label polygon can leave a faint mark.
- **Door kind (a): the doorway (door label pixels, which span the wall) painted in the median grey of the wall pixels within 3 wall thicknesses; the leaf and swing arc are inpainted away. Door kind (b): doorway + leaf + arc inpainted.** Own choice: leaf/arc = connected dark pieces (not wall/opening/fixture in the label) with >= 80% of their pixels within one door length of the doorway (a first version erasing every dark pixel in that zone wiped hatching and furniture).
- **R5 doorless-gap rule, final (user conditions): a straight contact line of length in [Lmin_door, Lmax_door] between two different spaces with wall at both ends; pair direct-contact rate < 10%; one side a bedroom, a bathroom, or an other_room >= p25 bedroom area; the other side not an other_room < p25 other_room area.** Reason (user): bedroom/bathroom openings without a door occur in only ~1-3% of real adjacencies; open-plan pairs and closets are left alone.
- **R5 cut-offs on the tuning plans only (`detect/gap_tune.py`, `outputs/stage2/gap_rule.md`): p25 bedroom = 1514 px, p25 other_room = 460 px (completion scale); shortest firing gap raised from 5 to 11 px to fit the budget.** Own budget reading: the tuning split's detector alone is already at 0.50 false alarms per intact plan (it was chosen at <= 0.5), so "total <= 0.4" cannot be met on tuning plans; instead R5 may add at most 0.4 - 0.31 (the reported held-out detector + C8 rate) = 0.09 per intact plan, measured on the tuning intact plans with R5 counted on its own (conservative). Result on tuning: 0.09 false alarms, door (b) recall 33% (6/18; the size condition alone gave 44%, the pair rule alone 72% at 0.23 false alarms).
- **Known limitation: on a drawing R5 depends on the parser's room types** (a bedroom parsed as living or hallway makes its open doorway look open-plan).
- **Check-set door (b) cases are doors R5 is meant for: closed pair with a bedroom or bathroom on one side, doorway >= the R5 minimum length** (own choice: tests R5's scope, not the parser's area estimates).
- **All evaluations run as Kaggle CPU kernels** (user: the laptop keeps running out of memory): `floorplan-stage2-eval-{doors,checkset,text}` (`kaggle/kernels/stage2_eval_*`, shared setup `src/stage2/kaggle_cpu.py`; checkpoints copied from the parser-v1, k2a and k1 kernel outputs to the paths in `configs/stage2/pipeline.yaml`). Laptop: unit tests + smoke tests on <= 3 items only.
- **Held-out detection on drawings (`detect/image_eval.py`): the 159 held-out val plans; 59 intact drawings for false alarms, the other 100 each get two separate omissions, kinds in turn over door (a), door (b), window, wall, railing, fixture, room (seed 5; random component of that kind; a kind the plan lacks -> the next).** Own choices: image level (erase on the drawing, then the pipeline's PARSE + CHECK, no fill), so parser errors count; door (a)/(b) use ANY door (not only those in C8's / R5's scope), with the in-scope subsets reported separately; a wall omission = a piece 5 wall thicknesses long of a straight run; a room omission = the walls and doors between a room (>= 2000 px) and its main neighbour.
- **Check set (`pipeline/check_set.py`): automatic scoring (found / restored / extra additions) plus my own verdict after looking at the figure.** Drawings in file order, first that fit each case kind. A wall piece = the middle 5 wall thicknesses of the first long straight run.
- **Text erasure for the text-dependence test (`parse/text_dependence.py`): dark (< 128) 8-connected pieces with bounding box <= 32 px per side (4 wall thicknesses), <= 400 px, >= 3 px on one side, not touching walls/openings/railings/fixtures (grown 2 px), inpainted.** Seen in the smoke test: room names are removed; the dashes of dashed lines go too (character-sized); text touching a wall stays. Parser at scale 1 (the parser set is already ~8 px walls), IoU pooled over drawings.
- **First run of the three kernels showed two erasure flaws; fixed and re-run (counts as the one retry).** (1) Door kind (a) on drawings whose walls are two thin outlines with a light inside: "median wall colour" was light, so the doorway still looked open (C8 found 0 of 15 cut-off rooms). Now the wall is drawn as a bridge (the wall mask closed across the gap) painted line by line with the median grey of that line's wall pixels just past the doorway, which continues outline walls as outlines and solid walls as fill. Leaf/arc = geometric bands (leaf line + quarter circle around the best of 4 hinge options, not across outdoor hatching) united with the earlier line-piece rule (thick arcs). Still imperfect for double-swing doors drawn as a box and for doors in angled walls (fallback fill). (2) Text erasure also removed thin non-text lines on small-scale drawings and left letter ghosts: now only pieces lying inside rooms (>= 95% on room labels), ink threshold 170; plus a control row (wall / door / window IoU original vs erased).
- **Image-level evaluation also records how much of each removed component the parser labels back as that class ("redrawn")**: in a diagnostic on one drawing the parser drew an erased window back at 89% and half of an erased wall piece; an erased appliance was not redrawn.

## Erasure v2: whole symbol removed (2026-10-19, user request after erasure_check review)

- User: the v1 erasure left most symbol ink (swing arcs lie outside the CubiCasa door polygon; shower hoses, sink smudges).
  New zones (user spec): door = opening + a square of side = door length on both sides of the wall; fixture / window =
  polygon box grown 40%. Kind (a) = kind (b) + wall drawn across.
- Own choices:
  - Inside a zone only the component's pixels and ink STROKES are inpainted, not the whole zone. Clearing the whole
    square left hourglass smears on grey / hatched fills. A stroke is >= 20 grey levels darker than the local
    background (closing, 15 px).
  - Straight runs longer than 60% (door zone) / 90% (symbol box) of the zone count as background texture (hatching)
    and stay. 90% for boxes because a fixture outline spans ~70% of its box.
  - Kept: all other labelled components and the arc zones of other doors.
  - A window is replaced by the wall drawn across it (an omitted window is a plain wall), as for door (a).
  - Railings, wall pieces and rooms keep the plain polygon erasure (not in the user's list).
- Known leftovers (erasure_check.png): a faint dashed arc and a dark blob on the hatched balcony of row 1; small grey
  smudges where strokes crossed hatching (rows 4-6).

## Parser v2 (launched 2026-10-19 10:08, kernel floorplan-stage2-parser-v2, T4 x2, one variant per GPU)

User: v2A "no text" (text erased on 100% of crops, text always erased at inference) and v2B "mixed" (50%), both
fine-tuned from v1 with C3, C6, C9, C10, C11, C12, C15, C16 at gradient-calibrated weights; no omission augmentation.
Own choices:
- Text eraser (src/stage2/parse/text_erase.py): character-sized pieces joined into words / lines (10 px gap); a group
  counts as text if >= 80% of its pieces lie on room labels and it has <= 6000 px of ink. All dark pixels in an
  accepted group's box go, except labelled components.
  Leftovers seen in text_erasure_examples.png: labels on tiled floors and labels touching a fixture stay. Short
  hatch / dashed strokes inside small closets can go too.
- v2A at inference: parse, find the text with that parse's labels, erase it, parse again (no ground truth at inference).
- Constraints at completion scale: each sample's softmax is shrunk by 3 / (8 x crop scale) (area), labels by nearest;
  samples with ignore pixels get no constraint loss.
  Weights: w_i = rho x median G_0 / median G_i over 20 batches, rho = min(0.05, 0.4 / 8), cap 10 x median (the
  Stage 2 rule), measured per variant at the start; saved in each run's constraint_weights.yaml.
- Fine-tune: lr 1e-4 (encoder 3e-5), 200 warm-up steps, batch 8, 1.85 h of training per variant.
- Checkpoint selection on 60 TUNING val drawings (configs/stage2/val_split/, same tuning / held-out definition as
  the detector): v2A by mIoU on text-erased drawings, v2B by the mean of normal and text-erased mIoU. The 159
  held-out drawings are used only for the final table (src/stage2/parse/eval_v2.py -> parser_v2.md).
- Step 0 of each run validates v1 on the same tuning set (baseline row in val_log.csv).

## Erasure v2 tweak (2026-10-19, user)

- User: fixtures and textured zones get a colour fill instead of cv2.inpaint (X smears); the faint dashed arc must go
  (lower ink threshold in door zones).
- Own choices:
  - Door zones use ink >= 10 grey levels (others 20).
  - Fill colour = LOCAL mean of the zone's non-ink pixels (17 px window, else 33, else the zone median) + Gaussian
    noise (sd = 1.48 x MAD of those pixels, <= 6). A single zone median painted white blocks, because a door zone
    spans both sides of the wall.
  - Textured zone (std of its remaining pixels > 12): after the colour fill, the stripes are continued by copying
    the pixel one zone-length away along the stripe direction. The source must be outside the filled / kept pixels,
    not a short ink stroke (no text or symbols copied) and in the same space label (never across a wall).
  - Flat zones keep inpainting.
- Remaining (erasure_check.png): a small mark in the hatching of row 4; a slightly paler patch in row 6.

## Pipeline parser = v2B (user, 2026-10-19)

- configs/stage2/pipeline.yaml and the CPU eval kernels now use parser v2B (mixed text erasure; held-out room-type mIoU
  76.3 normal / 64.3 text erased, v1 78.1 / 54.9; outputs/stage2/parser_v2.md).
- Caveat (user): the text eraser is inconsistent. It misses some labels (tiled floors, labels touching fixtures)
  and sometimes erases hatching / dashed strokes. Text-erased numbers are therefore approximate.

## C8 parts by footprint; sketch prior (user, 2026-10-19)

- C8: building parts = 8-connected non-background pixels (walls, openings, rooms, fixtures AND outdoor; separated
  only by background).
- Diagnostic (src/stage2/detect/c8_diag.py; local, 2 held-out cases + colorful-1261) found two other causes:
  - C8's score threshold: rooms < 2756 px (completion scale; tuned on label maps) are never reported. A cut-off
    room of 938 px was dropped.
  - Cut-off rooms reached through another door the parser drew (or a wall gap).
  The 2756 px threshold is re-tuned in step 5 (whole-CHECK budget on parsed tuning drawings).
- Sketch prior (src/stage2/complete/sketch_prior.py, configs/stage2/pipeline.yaml fill.sketch_prior):
  - Median sizes from the 212 tuning plans (configs/stage2/sketch_prior.yaml): door 13 px, window 20 px,
    toilet 5x9, sink 7x9, bath/shower 3x6, appliance 9x11.
  - Own choices: the wall line = the sketch's wall runs (>= 4 px) in the hole closed along their direction over
    24 px.
  - A door goes into an existing gap of that line if there is one (R5 / ink holes), else median length centred
    on the line. A window goes only on line pixels within 4 px of outdoor / background.
  - A fixture is a median box against the wall pixel nearest the hole centre, long side along the wall.
  - Example figure: outputs/stage2/sketch_prior_examples.png. Holes there sit at the true omission with its true
    type (the fill only).
- C8 diagnostic result (Kaggle CPU, 15 held-out cut-off cases + colorful-1261; outputs/stage2/c8_diag.md):
  - 8 below the 2756 px area threshold (rooms 259-1479 px);
  - 3 reached through another door the parser drew;
  - 2 with the erased door redrawn;
  - 2 flagged; colorful-1261 (multi-floor) flagged.
  None was caused by the per-part seeding. The footprint definition is kept as the user asked (no case changed by it).
- Door-ink check (src/stage2/detect/door_ink.py, rule "INK"): score = stroke pixels in the door zone (both sides)
  per px of door length, parsed walls / windows / railings / fixtures and the opening excluded. Example
  colorful-1148: erased door 0.05; intact doors 0.0 (2 of 17), 4.6-22.8 otherwise. Threshold tuned in step 5.
- Whole-CHECK re-tune (src/stage2/detect/check_tune.py): 40 intact + 80 x 2 omissions on parsed TUNING drawings.
  Grid: detector threshold x min blob, C8 min area, R5 on/off, ink threshold. Choice = best mean recall within
  0.4 false alarms per intact drawing. Written to configs/stage2/detector_threshold.yaml.

## Detector v2 and the 0.6 budget (user, 2026-10-19)

- Detector v2: data kernel floorplan-stage2-detv2-data (14:24) parsed the train drawings with parser v2B
  (intact + 2 image-level omissions each, erasure v2), plus the 212 tuning drawings for validation.
  Training kernel floorplan-stage2-detv2 (14:36, 2 h) warm-started from v1, 70% parsed / 30% label-level windows.
  Parsed tuning windows: mean F1 0.03 -> 0.36, intact windows with a false alarm 0.70 -> 0.19.
- Own choices:
  - parsing at scale 1 only (the parser set's walls are already ~8 px);
  - validation and checkpoint choice on the parsed TUNING drawings;
  - lr 1e-4, 100 warm-up steps.
- Ink check restricted to R5's pairs (door_ink.relevant): both faces rooms, closed pair, bedroom / bathroom /
  other_room >= 1514 px on one side, no closet on the other. The per-drawing guard still looks at all doors.
  Ink check alone at 1.5 on tuning drawings: 0.33 -> 0.05 false alarms, recall 12% -> 6%.
- Two-stage tuning: rules + ink first within 0.35, then the detector inside what is left of 0.6.
  - The first run let the tuning choose all doors for the ink check; re-run with the restriction enforced (user).
  - Chosen: detector v2 0.9 / 20 px, C8 >= 200 px, R5, ink < 3.0 on R5 pairs. Tuning: 0.57 false alarms, 19%.
    The detector uses 0.22, not 0.25: the next looser detector setting went over 0.6.
- Held-out (159 drawings): 0.64 false alarms, 15% mean recall. Cut-off rooms found 53%. Curve in
  outputs/stage2/check_curve.png / .md.
- Check set 3 / 12.
- Own housekeeping: the local smoke outputs of check_tune / check_curve (check_tune.md overwritten,
  check_curve.md / .png, made by me minutes before) were reverted / removed before the Kaggle runs.

## R6, R7, ink tiers, detector v3 (user, 2026-10-19)

- R6 "windowless exterior span" (src/stage2/detect/window_runs.py):
  - Exterior run = the room's boundary pixels that look straight through 1..T_wall + 3 px of wall only to outdoor /
    background; windows and doors break the runs.
  - Thresholds = 95th percentile of each type's longest run over 14,207 TRAINING rooms (configs/stage2/
    window_rule.yaml): living 104, bedroom 81, kitchen 51, other_room (>= 1514 px) 107 px.
  - Box: a median window (20 px) centred on the run.
- R7 "gap too narrow" (src/stage2/detect/narrow_gaps.py): wall runs >= 5 px, closed over Lmin_door; the gap spans
  >= 2 px across the wall.
- Ink tiers: tier 1 = R5's door pairs, tier 2 = other interior doors (threshold <= tier 1); exterior doors never;
  the guard still uses all doors.
- Tuning (0.6 budget, tuning drawings): detector v2 0.9 / 20, C8 >= 700, R5, ink 1.5 / 1.0, R6 x1.2, R7 off.
  0.55 false alarms, 20%.
  - R6 +0.6 points for +0.03 false alarms.
  - R7 nothing (off).
  - Tier 2 +3.0 points for +0.18.
- Held-out: 0.56 false alarms, 13%; R6 adds nothing there. Check set 3 / 12; tier 2 brings back case 7's false doors.
- Railings: 0% recall in every held-out run, so railing proposals are dropped from automatic CHECK
  (detector_threshold.yaml manual_only_types); railings only via manual mode (user rule). Evaluation tables still
  list the railing omissions (recall 0) for honesty.
- Detector v3 (drawing + parse): kernel floorplan-stage2-detv3, launched 20:01. 384 px crops at parser scale, 4 per
  version; ImageNet ResNet-34, 17 input channels; 1.9 h.
  - In CHECK (src/stage2/detect/detector_v3.py): parser-scale grid rebuilt from the completion map (x 8/3);
    probabilities shrunk back, so the same proposal thresholds apply.
  - The final tuning chooses v2, v3 or both (own thresholds each).
- Detector v3 result (512 validation crops of parsed TUNING drawings, outputs/stage2/detector_compare.md):
  - mean F1 v1 0.05 / v2 0.07 / v3 0.56;
  - intact crops with a false alarm 88% / 48% / 11%.
  v2 is shown on completion-scale maps rebuilt from the crop, so this setting disadvantages it slightly; on its own
  128-px windows it reached mean F1 0.36.
  Caveat (own): v3 sees the erased drawing, so it may partly learn the erasure's traces (smoothed / inpainted
  patches) rather than missing components. Its 97% on erased walls is suspicious. Real omissions leave no such
  traces; a check with fake erasures of empty areas would tell.
- Final round (budget 0.6, tuning drawings): detectors v2 (0.5 / 40 px) + v3 (0.5 / 20 px), C8 >= 700, R5,
  ink 1.5 / 1.0, R6 x1.2, R7 off.
  - Tuning: 0.35 false alarms, 58%; v2 alone 25%, v3 alone 54%.
  - Held-out: 0.41 false alarms, 59%; the detectors alone give 58% at 0.17.
  - Check set: 4 / 12 correct + 3 partly (automatic 5 / 12).
- Railings: right-type recall still 0% (6% only via other-type boxes), so they stay manual-only.
- Not yet checked: how much of v3's gain comes from erasure traces. Proposed control: fake erasures of empty room
  areas on intact held-out drawings; v3's firing rate there.
- Erasure-trace control (outputs/stage2/fake_erasure.md; 40 intact held-out drawings, fake erasures of
  non-components). v3 fired on:
  - 0% of erased words;
  - 2% of blank floor patches (baseline 2%);
  - 23% of erased furniture drawings (3 of 13; baseline 0%).
  v2 never fired. Real omissions: 58%. So v3 reacts somewhat to cleaned-patch traces (furniture), but most of its
  recall is not explained by traces. Kept as is (user: stick with the final CHECK); caveat noted for the report.

## Overnight 2026-10-19/20: Stage 1 final test run (task 1)

- The one Stage 1 test run, frozen Day 7 configurations (configs/app.yaml); the TA's new scope replaced the
  approval step (logged earlier).
  - src/full_eval.py and src/arch_eval.py got `--benchmark val|test` (default val).
  - src/selection.py `--benchmark test` (test_main) scores the saved test predictions with the same metrics, builds
    the FROZEN size-adaptive hybrid (each size bin's cases from the fixed t0 400 / 600 / 700 hybrid, no
    re-selection), runs the paired bootstrap against matched U-Net, refine-600 and hybrid 600, and writes
    outputs/stage1_test.md with a val-vs-test table.
- Own choices:
  - The test benchmark was uploaded as a separate small private dataset, floorplan-test-benchmark (3 MB); the
    kernel checks its sha256 against configs/benchmark_checksums.json.
  - Sampling (GPU kernel floorplan-stage1-test, launched 23:24) is separate from scoring (CPU kernel
    afterwards), so the GPUs don't idle while CPU scoring runs.
  - Seeds per setting as in full_eval (100 + position in the --only list).
  - Smoke test: 20 test cases with 3 sampling steps on the laptop, the whole pipeline end to end.
  - "> 2 points" flag = |test - val| > 2 percentage points on shares / IoU / F1 (x100) and on validity %; counts
    (conn_err, room_count_err, dead ends) are shown raw and not flagged.
- Local copies of data/benchmarks/{val,test}_benchmark.* were placed in the worktree's (git-ignored) data/ folder.

## Overnight 2026-10-19/20: generous flagger + judge (task 2)

- Ink tier 2 (other interior doors) switched off (user): it fires on drawings whose doors have no arcs
  (check-set case 7).
- Generous flagger + U-Net judge (src/stage2/detect/judge.py; kernel floorplan-stage2-judge).
  - Generous setting (tuning drawings, best recall within 1.2-2.0 false alarms): detectors v2 + v3 at 0.3 / 10 px,
    C8 >= 0 px, R5, ink 1.5 (R5 pairs), R6 x1.0.
  - Judge: k2a 16-class U-Net, unforced, no sketch prior. Score = mean P(type) in the masked box - parse share.
  - Logistic on [judge, ink / 10 (doors), is_door] fitted on 588 tuning flags; threshold 0.5 for <= 0.4.
- Own choices:
  - the generous grid (detector 0.3-0.5, C8 0 / 200 / 700, ink 1.5 / 3.0, R6 1.0 / 1.2);
  - one logistic for all types;
  - room flags judged on the wall class;
  - winning rule = higher held-out mean recall at no more than +0.1 false alarms.
- Held-out (159 drawings):

  | CHECK | false alarms | mean recall |
  |---|---|---|
  | current final (ink tier 2 off) | 0.32 | 58% |
  | generous, no judge | 1.46 | 70% |
  | generous + judge | 0.42 | 34% |

  The judge's score hardly separates true omissions from false flags (judge_scores.png: everything near 0.5 except
  doors, where it mostly repeats the ink check).
- Decision (user rule): the current final stays; generous + judge is logged as a tried ablation (use_judge off).
- Note: the generous flagger alone reaches 70% at 1.46 false alarms, a possible "high-recall mode" for the app.
- Stage 1 test run result (outputs/stage1_test.md).
  - Test, size-adaptive hybrid: tolerant mIoU 0.846, validity 88.0%, connectivity error 0.011.
  - Test, matched U-Net: 0.850, 81.0%, 0.085.
  - No metric of any method differs from val by more than 2 points.
  - Paired bootstrap conclusions as on val: size-adaptive vs matched U-Net better on connectivity and validity,
    worse on tolerant mIoU; vs refine-600 better on all three.
  - First launch stopped at its own safety check: the kernel compared the npz FILE hash, but
    configs/benchmark_checksums.json holds make_benchmark's ARRAY checksum. Fixed and relaunched; no GPU work was
    done by the first attempt.
  - GPU session ~1.5 h (sampling) + CPU scoring kernel.

## 2026-10-20: omission metadata, eta tuning, end-to-end validation

- Omission metadata (deliverable 5; src/stage2/data/omission_meta.py): every image-level omission
  (image_eval.omit, check_set.make_case, hence check_tune / check_curve / judge / e2e) records type, location,
  position, size bin, size %, centre distance.
  - Stage 1 constants read from configs/corruption.yaml (edge cutoff 0.55; size bins 2.7 / 7.7 %), not copied.
  - Own choices: building = every pixel that is not outdoor / background (Stage 1: wall + door + window + room);
    size = the omitted component's own pixels (Stage 1 used the padded mask box); "touches" = 4-neighbour.
  - The label-level simulator (omissions.simulate_one, older bins 0.5 / 2 %) is left as it was: its tables are
    already reported.
  - breakdown() / breakdowns() split any per-case table by the four fields; image_eval's report and e2e_val use it.
- eta (guidance strength), user rule, kernel floorplan-stage2-eta (GPU T4, 09:23-10:02, ~0.65 GPU h):
  150 holes from the TUNING val plans (25 each: door, window, wall, railing, fixture, room; region left out:
  not a component), fast hybrid with the sketch prior, same seed per hole for every eta; raw sampler output scored.
  - eta 0 / 0.1-3: tolerant mIoU 71.7 / 71.6, violations 5.35 / 5.32, validity 59 / 59 %. 0.1, 0.3, 1 and 3
    give IDENTICAL output; the 0.4% of hole pixels that differ from eta 0 come from guide_x0's clamp(-1, 1),
    not from the constraint gradient.
  - Why: the calibrated constraint losses' gradient through softmax(x0 / 0.2) is ~1e-8 - 1e-6 per pixel on
    the near one-hot x0 of the last steps; guidance only acts from eta ~1e5 (laptop check, then the extra rows):
    3e5 -> 70.0 / 6.05, 1e6 -> 66.8 / 8.53, 3e6 -> 62.2 / 15.04 (worse on both).
  - Choice (user rule: fewest violations within 1 mIoU point of eta 0; ties -> higher mIoU, smaller eta):
    eta = 0.1, written to configs/stage2/pipeline.yaml. Own choice: the extra rows (1e5-3e6) were diagnostic
    only and not eligible, since the user named the grid. Same answer either way.
  - Note for the report: in this setup x0 guidance is effectively off; the verifier (best-of-N) and the
    training-time constraints carry the constraint work. Guidance costs ~2.4x fill time on GPU (1.55 vs 0.64 s
    per hole).
- End-to-end validation (kernel floorplan-stage2-eval-e2e, CPU, 10:04-10:11; outputs/stage2/e2e_val.*):
  12 held-out val drawings minus the check set, fixed random order (seed 20), 2 per kind, one omission each
  via image_eval.omit (erasure v2), run_pipeline(fast=True) with eta 0.1.
  - Own choices: "fixed" = the added right-class pixels touching the omission (grown 3 px) have IoU >= 0.3
    with it (room: its wall / door band vs added walls); mIoU / validity inside the omission box + 4 px at the
    parsed map's scale, ground truth resized nearest; "before the fill" columns added so the fill's effect shows.
  - Found 11 / 12, right type 9 / 12, fixed 3 / 12 (+3 where the parser had redrawn the omission), 2 false
    additions, tolerant mIoU 56.8 -> 59.2, validity 33% -> 8%, 27 s per drawing.

## 2026-10-20 (afternoon): fixes from the end-to-end figure

- eta = 0 (user): guidance tested, no measurable effect at safe strengths (0.1-3 identical to 0), 2.4x slower.
  configs/stage2/pipeline.yaml fill.eta_guidance = 0.
- Symbol already in the parse (user; run.existing): a door / window / railing / fixture flag with a location
  whose type the parse already has at the flag (>= half of the component inside the box + 2 px, or under the
  flag point) -> "<type> symbol missing in the drawing -> added", drawn red on the overlay (result "marked"),
  no fill. Own choices: C8-only flags (no location) and wall / room flags never use it; the nearest component
  to the flag point is the one marked. Marked components count as additions for false adds.
- "Fixed" (user): a flag overlaps the omission AND the final plan has the right class there with IoU >= 0.3,
  whoever drew it. Own choices: door / window / fixture = whole components of that class touching the
  omission (grown 3 px); wall / room (class wall) = only wall pixels within 3 px of the omission (walls are one
  connected network, whole components would make IoU meaningless). The earlier definition stays as
  "fixed by addition". The check set reports both: correct (old) and correct (new) = fixed and no false adds.
- Door at the flag (user; sketch_prior.place_opening, run.restrict): flag point = detector heat peak (new
  Candidate.peak, kept through merge), else box centre; none for C8-only flags (old rule kept). The prior
  uses the wall line passing nearest the point, only its part nearest the point, centred on it; a gap on the
  line -> the gap part nearest the point. The restriction keeps the new door / window nearest the point
  (largest without a point).
- Wall vs door gaps (user; src/stage2/detect/flag_gap.py): applied to door flags from R5 / INK / detector.
  Own choices: (a) breaks shorter than Lmin_door also become "wall missing" (no door fits; same as R7);
  (b) wall flags with a door-sized gap stay walls and room flags are not touched (else every 15 px
  wall-piece omission in the check set would become a door, and fix 4 governs room flags); (c) measurement:
  wall runs >= 5 px, closing over 2 x Lmax_door, break 2-6 px across the wall, within 3 px of the flag point,
  length = longest run on one row / column (not the box: a diagonal wall gives a long box). A wall gap is
  painted as wall into the sketch.
  On the two drawings named: drawing 7's erased exterior wall was parsed as a 14 px WINDOW (no gap to measure;
  the flag is a detector "room"); drawing 8's 15 px erased wall was mostly redrawn as wall by the parser,
  leaving a 7 px door-sized break -> door. So the rule, as specified, cannot change either; wall pieces of
  5 wall thicknesses (~15 px) are door-sized by construction.
- Room holes (user): walls + at most one door (nearest the point, clipped) + ONE room type, the most common
  among the hybrid's changed room pixels; other changed room pixels revert to the parse.
- Validity diagnosis (user): e2e_val now lists, for drawings where the fill added violations, each added rule
  and whether the raw hybrid output already had it (hybrid) or only the restricted plan (restriction).
  Laptop check on the old run: drawing 2's extra violations (C7 slivers) came from the restriction: the
  hybrid moved the wall line under its door, the restriction kept the door and reverted the wall. Simple fix
  applied: a kept door also keeps the hybrid's new wall pixels within 2 px of it (run.door_walls; on drawing 2
  13 -> 12 violations, raw 11: partial). Drawing 7's came from the hybrid itself (not changed).
- Re-run: same 12 e2e drawings (tag _v2) + the 12-drawing check set (tag _v2), kernel floorplan-stage2-eval-e2e
  v2 (CPU). Results below.
- Run v2 (kernel version 2) had the gap rule BEFORE the existing-symbol check: on e2e drawing 4 the door the
  parser had redrawn was measured with its surroundings as a 53 px gap and closed with wall (tol. mIoU 97 -> 58).
  Own fix: the existing-symbol check runs first (a symbol already at the flag is never a gap); re-run v3.
  Both kept: outputs/stage2/e2e_val_v2.*, check_set_v2.* (record) and *_v3.* (current).
- Results v3 (same 12 e2e drawings; old -> new): found 11 -> 11, right type 9 -> 9, fixed by addition 3 -> 3,
  fixed (new definition) 9 / 12, false adds 2 -> 2, tol. mIoU after 59.2 -> 60.7, validity after 8% -> 8%,
  27 -> 15 s per drawing (eta 0 + no fill where the symbol is already parsed). Check set: correct (old
  definition) 5 -> 5 / 12, correct (new definition) 6 / 12.
  - Caveat on the new definition: e2e drawing 8 (wall erased) counts as fixed because the parser redrew most
    of the wall (wall IoU 0.31), although the pipeline itself reports a false "door symbol missing" there.
    Check-set case 7 drops from correct to not: its 4 marked ink doors away from the omission now count as
    false additions (they were silently skipped before).
  - Validity diagnosis v3: the rules the fill adds (drawings 1, 7, 10: C7, C9, C13, C14, C16) all come from the
    raw hybrid output, none from the type restriction; nothing further to fix simply.

## 2026-10-20 (evening): parse vs drawing double check (window, wall), room doors, CHECK re-tune at 0.4

- New src/stage2/detect/symbol_ink.py (doors: door_ink.py unchanged).
  - Window: the user's "thin parallel lines" profile is computed (runs = dark runs across the band) and
    reported in the calibration, but the rule uses d_wall = mean |grey profile across the window - grey profile
    of the parsed wall on the same line past its ends| / 255. Own choice: erased windows are painted in the
    drawing's own wall style (erasure.paint_wall), and many styles draw walls as two thin lines, so "solid dark"
    would miss outline-style walls; "looks like the neighbouring wall" covers both. Gap = stroke share < WG.
  - Ink = >= 20 grey levels below a 31 px background closing, or darker than 110 (thick solid walls are wider
    than the closing kernel; with 15 px only, solid-wall drawings showed 24% "uninked" walls).
  - Wall: uninked parsed wall runs (no stroke within 1 label px), length >= WL.
  - Guard 40% per drawing and type (windows: share of windows lacking the symbol; walls: share of parsed wall
    pixels with no ink).
  - Flags carry the parsed component (Candidate.mask, kept through merge for the same type); the pipeline
    marks it ("window symbol missing in the drawing -> added"; no-ink windows: "window missing (a gap in the
    drawing) -> added") and draws it red; scored as fixed by the existing definition.
- Room holes: the restriction keeps a door only if both faces are room types (never on an exterior wall).
- CHECK re-tune (kernel floorplan-stage2-eval-tune v7, CPU; outputs/stage2/check_tune.md): budget 0.4 as asked
  (the previous tuning used 0.6; its held-out result was 0.41). Own choices: rules first within 0.25 (detector
  share 0.15, was 0.25 of 0.6), ink tier 2 kept off (user, 2026-10-19), window grid d_wall < 0.03-0.12 x gap
  ink < 0-0.05, wall runs >= 4-13 px.
  - Calibration: parsed windows at window omissions d_wall median 0.084 vs 0.242 on intact windows (16 of 23
    window omissions still had a parsed window); uninked wall runs at wall omissions: 1 of 23 (the parser rarely
    draws a wall over an erased gap).
  - Chosen: same detector / C8 700 / R5 / door ink 1.5 / R6 x1.2, + window ink d_wall < 0.08, ink < 0.02, + wall
    ink >= 4 px: 0.30 false alarms, 57% mean recall on tuning. Window check +1.2 points for +0.05; wall check
    +0.0 for +0.05. Own choice: wall check switched off (a tie on recall -> fewer false alarms, the tuning's own
    tie-break) -> 0.25 false alarms.
- Results (CPU kernels eval-doors v9, eval-e2e v4):
  - Held-out table (outputs/stage2/detection_eval.md, last section; same 59 intact + 200 omissions as before):
    CHECK before the double checks (current config: ink tier 2 off) 0.32 false alarms / intact drawing, 58% mean
    recall; + window ink (final) 0.59, 60% (window 62 -> 66%, wall 72 -> 76%, railing 6 -> 11%); wall ink alone
    (diagnostic) 0.46, 60%. The window check cost +0.05 false alarms on tuning but +0.27 on held-out: it does not
    generalise and takes the CHECK over the 0.4 budget. Not switched off here (settings are chosen on tuning,
    never on held-out); left to the user.
    (The earlier "final" held-out row, 0.41 / 59%, still had ink tier 2 on.)
  - e2e, same 12 drawings, v3 -> v4: fixed 9 -> 9, false adds 2 -> 3 (two window-ink marks on drawing 1),
    drawing 7 now flagged "window missing (a gap)" (wrong type: a wall was erased), validity after 8% -> 17%.
  - Check set v3 -> v4: correct (new definition) 6 -> 6 (case 5 window now found and marked; case 8 gets one
    false window mark), correct (old definition) 5 -> 4.
  - Batch 2 (12 fresh held-out drawings, seed 21, batch 1 left out; outputs/stage2/e2e_val_batch2.*): found 8,
    fixed 8 (2 by addition), 5 false additions on 4 drawings, tolerant mIoU 76.5 -> 74.5, 19 s per drawing.
    Verdicts in the file. Own note: these drawings were in the held-out table's aggregate (all 159 held-out
    plans are) but never looked at one by one or used to choose anything.

## 2026-10-20 (night): large omissions, batch 3, size-balanced test mix

- New src/stage2/data/erasure_large.py; kinds "room (whole)" and "region" in image_eval.omit (expected type "room").
  Own choices:
  - Room (whole): rooms of type bedroom / bathroom / kitchen / other_room at 2-15% of the building (as asked).
    Interior walls = wall pixels within 2 wall thicknesses of the room and not within 2 wall thicknesses of
    outdoor / background (the outer building walls stay). Its doors (in its walls) are cleared with their arcs;
    exterior doors and its windows are closed with wall in the drawing's style (as door (a) / window omissions:
    the outer wall stays continuous). Then every remaining stroke inside the merged room, its name text
    included, is filled with the local paper grey (Telea inpainting alone left smears and half fixture symbols).
  - Region: rectangle, aspect 0.6-1.6, 5-25% of the building inside it, needs >= 100 px of interior walls /
    doors (else up to 30 retries), the same erase order and final stroke clearing.
  - Scoring: "found" / "fixed" on the erased components (walls, openings, fixtures; class wall, local IoU as
    for room omissions); the size bin on the whole area (the room + its walls / the rectangle in the building).
  - Examples figure from TUNING drawings (outputs/stage2/erasure_large_examples.png), so batch 3 stays fresh.
- Batch 3 (e2e_val --batch 3): held-out order as before, batches 1 and 2 left out, seed 22; 4 room (whole),
  4 region, 4 small (a random small kind per drawing, the next if the drawing lacks it).
  Realised sizes: room (whole) 1 medium + 3 large, region 4 large, small 4 small.
- Test mix (configs/stage2/test_eval.yaml, src/stage2/detect/omission_mix.py): plan_bins cycles small / medium /
  large in equal thirds; small = the image_eval kinds kept only if measured small; medium = room (whole) at
  2.7-7.7% or region at 5-7.7%; large = room (whole) at 7.7-15% or region at 7.7-25%; each case checked against
  its own measured size bin; a drawing where nothing fits takes the next bin (realised counts to be reported).
- Batch 3 results (kernel eval-e2e v5, CPU; outputs/stage2/e2e_val_batch3.*): found 7 / 12, fixed 2 / 12 (large:
  1 of 7; medium 0 of 1; small 1 of 4), 10 false additions (6 window-ink marks), tolerant mIoU 44.8 -> 44.9,
  15 s per drawing. Large omissions get flags but of small types (sink / window / door); there is no "region
  missing" type and the fill windows are <= 100 px. Erasure caveat: on a grey-fill drawing (batch 3 #2) the
  room (whole) erasure left a visible grey smudge.

## 2026-10-20 (night 2): large holes - R8 merged-room rule, three large fills (B1-B3), comparison

- Part A, R8 (src/stage2/detect/merged_rooms.py): a parsed room instance is "region missing" when its area > mult x
  its type's p99 (R1 table) or it holds fixtures no single room type takes (C15 table). Own choices: other_room is
  dropped from the C15 sets (it accepts every fixture, so no intersection would ever be empty); a fixture belongs
  to a room if >= half of it lies in the room grown 2 px; box = the room's bounding box + 1 px; score = area / p99
  (>= 1), so region flags win merges with overlapping small flags.
- CHECK re-tuned with R8 (kernel eval-tune v8, CPU; outputs/stage2/check_tune.md), budget 0.4. Own choices: the
  tuning omissions now cycle over the small kinds AND room (whole) / region (so R8 can earn recall); grid pruned to
  stay < 1 h (R7 and wall ink off: both +0 before; window ink off or at 0.08 / 0.02). Because the omission set
  changed, the whole CHECK was re-chosen: detectors v2 0.6 / 20 px + v3 0.5 / 20 px, C8 >= 400 px, R5 OFF, door
  ink < 2.0, R6 x1.5, window ink 0.08 / 0.02, R8 > 1.0 x p99 + mix: 0.35 false alarms / intact tuning drawing,
  72% mean recall (room (whole) 89%, region 76%). R8 adds +1.1 points for +0.05 (alone it finds 12% of regions,
  6% of whole rooms). The window-ink decision (held-out cost) is still the user's.
- Part B, src/stage2/complete/large_fill.py: B1 big window, B2 coarse-to-fine (frozen Stage 1 hybrid, configs/app.yaml
  settings, then 16-class refinement at t0 150 / 300), B3 tiled (128 px, 32 px overlap, most-known-context first).
  Own choices: B1 / B3 use the Stage 1 size-adaptive t0 (400 / 600 / 700 by the hole's share), not t0_fixed 300
  (chosen on small holes); 16 -> 6 mapping: railing -> wall, fixtures -> room; Stage 1 hole = its frame pixels
  covered >= 25% by the hole; region holes = the flag's box. Restriction for region holes: everything kept except
  doors that do not join two interior spaces. Pipeline: region flags and manual boxes > max_hole (100 px) go to
  fill.large_strategy (provisional: tiled); checkpoints.stage1_unet / stage1_diffusion added to pipeline.yaml.
  Smoke (1 case, manual): B1 runs on a 256 px window (attention fine at that size).
- Part C results (kernels large-fill v1 (manual) and large-fill-auto v1, CPU; outputs/stage2/large_fill_compare*.md,
  large_fill_compare.png): manual (hole given): fixed B1 5 / 20, B2 3 / 20 (t0 150 and 300), B3 6 / 20; tolerant
  mIoU 27.9 -> 34.5 / 31.7 / 33.9; validity 15% / 0-5% / 15%; 21 / 42 / 18 s per hole. Auto: R8 fired on 0 / 20
  (merged rooms stay under p99: 0.64 x on batch 3 #4), so all strategies are identical (fixed 2 / 20).
  Own choices: the figure shows the first 2 cases of each group (no selection) and B2 at the t0 with the better
  manual mean (150); fill.large_strategy stays "tiled" (most fixed, fastest). Medium cases come from
  omission_mix.omit_in_bin (7 measured medium of 20).
- Part D: none of B1-B3 is clearly good -> B4 plan (fine-tune at 256 px with large holes, ~4 GPU h) written in
  outputs/stage2/large_fill_compare.md, not launched.

## 2026-10-21: R5 forced on, R8 v2, B4 launched, final test notebook

- R5 forced ON in CHECK (user): it targets the TA's "every room must have a door" case (a doorway left open between
  rooms that need a door). check_tune.R5_GRID = (True,); the rest re-tuned around it within 0.4. Tuning v9 (R8 v2 first
  grid): same detectors, C8 >= 700, door ink 1.5, R6 x1.5, window ink, R8 mix only: 0.30 FA, 71% mean recall; R5 costs
  +0.05 FA for +0.0 recall on the tuning omissions (kept by the user's decision).
- R8 v2 (src/stage2/detect/merged_rooms.py): relative to the drawing (user): area > k_type x the median of the OTHER
  rooms of its type, or > k_all x the median of all other rooms when the type appears once; or the fixture mix. Own
  choices: the room's own area is left out of both medians; rooms < 50 px ignored; box margin 4 px; score 2 + ratio
  (wins merges). First grid (1.5-2.2 x type / 2.0-3.0 x all) cost 1.2-1.6 false alarms per intact drawing alone
  (real living rooms / halls are several times the median room) -> tuning kept only "mix". Extended grid (own
  choice): up to 4 x type / 5 x all, and a variant whose size test skips living rooms and hallways; re-tune v10.
- Region flags: the whole flagged room's box (+ 4 px) goes to the large-hole fill (run.is_large), not to a
  single-type restricted fill (unchanged from 2026-10-20, now with the margin).
- B4 launched 2026-10-21 00:57 (kernel floorplan-stage2-b4 v1, T4 x2): GPU0 U-Net, GPU1 constrained diffusion, both
  warm-started from the CURRENT checkpoints (k2a_unet, k1_constrained best.pt) under new names b4_unet / b4_diffusion.
  Own choices: 256 x 256 windows, 50% large holes (windows.sample_large_window: whole room 2-15% / region 5-25% of the
  building, the erasure_large definitions at label level; hole = box + 2-6 px) + 50% the original small holes;
  batch 8 per GPU (4x the pixels of 128 px), lr 5e-5 (U-Net) / 2e-5 (diffusion), warm-up 100, cosine over the time
  budget; same 14 constraints and weights; checkpoint every 15 min; in-training val on 64 mixed val windows (seed
  1234). Hard stop: 1.65 h of training per run + a watchdog at 1.85 h after the kernel start (< 1.9 h, < 3.8 GPU h).
- Pipeline device: STAGE2_DEVICE (default cpu) moves the parser and the fill models to the GPU (detectors stay on the
  CPU); samplers get a generator on that device (run.generator). CPU results unchanged (checked on e2e drawing 10).
- Large-hole models: fill.large_models (current / b4) chooses which 16-class models the large-hole fill uses; B4
  checkpoints at checkpoints.b4_*.
- Final test notebook (prepared, not run): kaggle/notebooks/stage2_final_test (src/stage2/final_test.py, two shards
  on two T4s, per-case jsonl = checkpoint / resume, report); steps in docs/final_test_notebook.md; dataset builder
  src/stage2/package_final.py (checkpoints + offline wheels). Own choices: counts 40 / 40 / 40 fill windows, 60 intact
  + 120 omitted CHECK drawings, 60 end-to-end drawings (configs/stage2/test_eval.yaml); evaluation sampling (N = 4,
  DDIM 50); the ablation runs on the FILL with the hole given (same sketch and seed, constrained vs control
  diffusion), with paired bootstrap CIs. The test data come from the floorplan-stage2-data notebook output.
- Tuning v10 (extended R8 v2 grid; outputs/stage2/check_tune.md): no size variant fits the budget. Alone they cost
  1.6 (1.5 x type / 2 x all) down to 0.30 false alarms per intact tuning drawing (4 x type / 5 x all, size test
  skipping living / hallway: finds 50% of whole rooms, 35% of regions); the rules get <= 0.25, so the choice keeps
  R8 = fixture mix only (+1.1 points for +0.05). Chosen CHECK: detectors v2 0.6 / 20 + v3 0.5 / 20, C8 >= 700,
  R5 on, door ink 1.5, R6 x1.5, window ink 0.08 / 0.02, R8 mix: 0.30 FA, 71% mean recall on tuning.

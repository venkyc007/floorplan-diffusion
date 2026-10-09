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

## Tooling

- **Always pass `encoding="utf-8"` when Python reads/writes text files** — Windows defaults to cp1252; a write with "−" failed and emptied this file once (rebuilt from the session record).

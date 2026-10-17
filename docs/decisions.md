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

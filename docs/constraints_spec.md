# Constraint library: spec (Stage 2, 16 classes)

Written for Claude Code to implement. Save as `docs/constraints_spec.md`.

**Core idea.** Every constraint is **one function** that returns a per-pixel violation map `V` (shape B×1×H×W, values ≥ 0). That single function is used in four places:

| Use | How |
|---|---|
| Training loss | weighted masked mean of `V` over region `M` (§2, §3) |
| Sampling guidance | gradient of the loss with respect to the predicted clean plan x̂₀ during DDIM sampling (§5) |
| Detection | threshold `V` on the parsed full plan to get candidate masks; the `V` maps are also inputs to the learned detector (§6) |
| Verification and metrics | count thresholded violations, prediction vs ground truth (§5, §7) |

---

## 0. Conventions

**Classes (16):** wall, door, window, railing | living, bedroom, kitchen, bathroom, hallway, other-room | outdoor, background | toilet, sink, bath/shower, appliance.

**Groups** (each is a sum of class probabilities):
- `W` = wall, `D` = door, `N` = window, `RL` = railing
- `S` = all interior spaces (living + bedroom + kitchen + bathroom + hallway + other-room)
- `OD` = outdoor, `BG` = background, `OUT` = OD + BG
- `F` = all fixtures (toilet + sink + bath + appliance)
- `PASS` = S + OD + D (walkable pixels)

**Inputs**
- `P` = softmax(x̂₀ / τ) with τ = 0.2 for the diffusion model; softmax(logits) for the U-Net; one-hot `Y` for ground truth.
- `hole` = the inpainting mask. `M` = dilate(hole, 3 px). Losses are averaged over `M` only.
- Compute all constraint losses in **fp32**, even under autocast.

**Scale.** Plans are normalised so the median wall thickness is 3 px. **Set all size constants from ground-truth percentiles in the training data, not hard-coded metres.** For example, `Lmin_door`/`Lmax_door` are the 2nd/98th percentiles of door length in px. The constants used below are:
- `T_wall`, `T_wall_max`: wall thickness, typical and maximum
- `Lmin_door`, `Lmax_door`, `Lmin_win`, `Lmax_win`: opening lengths
- `k_c`: minimum width of class c
- `k_clear`: door clearance distance
- `k_fix`: fixture-to-room search distance

Write the constants to `configs/constraint_constants.yaml`, with the percentile used for each.

**Operators** (`src/models/constraint_ops.py`; all differentiable, all built on `max_pool2d`, whose padding acts like −inf, so there are no border artefacts)
- `shift(X, dy, dx)`: translate with zero padding.
- `dil(X, k)` = max_pool2d(X, k, stride 1, pad k//2) (soft dilation)
- `ero(X, k)` = −dil(−X, k) (soft erosion)
- `opn(X, k)` = dil(ero(X, k), k) (soft opening: removes anything thinner than k)
- Line versions with (1×L) and (L×1) kernels: `dil_h, dil_v, ero_h, ero_v, opn_h, opn_v`.
- `flank_h(X, L)` = min( max of X over the L pixels to the left, max of X over the L pixels to the right ). `flank_v` is the same for up/down. Meaning: "is there X on both sides along this axis".
- `sum_h(X)` = shift(X,0,+1) + shift(X,0,−1); `sum_v` likewise vertically.
- `rec(seed, passable, K)`: soft flood fill (geodesic reconstruction). `r = seed·passable`, then K times `r = min(passable, dil(r, 3))`. Use K ≈ 96 for a 128-px window. Wrap every 16 iterations in `torch.utils.checkpoint` to save memory.
- `sobel_x`, `sobel_y`.
- `smax(X, region)`: smooth max over a region via log-sum-exp (temperature 0.1).

**Two modes for every constraint**
- **absolute**: uses V(P) alone. Used at inference (guidance, detection, verifier).
- **GT-hinged**: `L = relu( mean_M V(P) − mean_M V(Y) )`. The model is only penalised for being *worse than the real plan*. Used in training wherever real plans sometimes break the rule. Gate G2 (§4) decides the final mode for each constraint from data; the modes listed below are starting defaults.

Status labels:
- **HAVE**: already in the current model.
- **RE-ADD**: built in Stage 1 and dropped after the overfit test because its weight was ~100× too strong (weights were set by loss value, not gradient size). Now re-added with calibrated weights.
- **NEW**: not a loss before. Some existed as validity metrics only.

---

## 1. Catalogue

### Walls

**C1 Walls connect (no dead ends or breaks)**: HAVE, GT-relative
- Training: clDice(W, Y_wall) on M (soft skeleton, as in Stage 1).
- Map form for inference: soft-skeleton endpoints of W that do not touch D, N or OUT.

**C2 No floating wall pieces**: NEW (was validity rule 2), absolute
- `V = relu( W − rec(W·(1−hole), W, K) )`. Wall predicted in the hole must connect to known walls.
- Only defined when a hole exists (fill and guidance), not at detection time.

**C3 Walls meet at 90° (Manhattan)**: RE-ADD, GT-hinged (CubiCasa has some angled walls)
- `V = abs( sobel_x(W) · sobel_y(W) )`, normalised by its maximum on a straight-wall toy.
- Axis-aligned edges have a gradient in one direction only, so the product is about 0.

**C4 Wall thickness in range**: NEW, GT-hinged
- `V = (W − opn(W, 3))` [too thin, slivers] `+ ero(W, T_wall_max + 1)` [too thick].

**C5 Walls continue straight into the hole (alignment)**: NEW, absolute
- Precompute from the known region (non-differentiable, constant per sample): the continuation lines of known walls that run into the hole, length 2·T_wall.
- `V = 1 − (W + D + N + RL)` on those pixels. A wall may continue as a wall or end in an opening.
- Only defined when a hole exists.

### Spaces

**C6 Different spaces never touch directly (closure; also covers an open building envelope)**: HAVE (generalised from 1 room class to 6 room types + outdoor + background), absolute
- `V = Σ over the 4 neighbour offsets δ, Σ over pairs a ≠ b in {living, bedroom, kitchen, bathroom, hallway, other, outdoor, background} of P_a · shift(P_b, δ)`.
- **Exclude pairs that touch directly in real plans** (for example an open kitchen–living area), using CubiCasa statistics: exclude any pair whose direct-contact rate exceeds 1% of its total adjacency. List the excluded pairs in the config.

**C7 Minimum width per space and fixture (no slivers, no 1-px rooms)**: NEW, GT-hinged
- `V = Σ_c relu( P_c − opn(P_c, k_c) )`, with k_c per class from GT percentiles of inscribed width.

**C8 Every room is reachable through doors** (the TA's "every room must have a door"): NEW (was validity rule 6), GT-relative
- Training:
  - `R_gt = rec(spaces on the window border, PASS(Y), K)` (what is reachable in the real plan);
  - `seed = R_gt · (1 − hole)`;
  - `V = S · R_gt · (1 − rec(seed, PASS(P), K))`.
- Inference (absolute): seed = the largest connected walkable component of the full plan (or the pixels next to exterior doors).
- Gradients flow into the door probabilities on blocked paths, which teaches the model to add a door where a room is cut off.
- **Must be trained together with C1 and C6.** On its own, C8 can be "satisfied" by deleting a wall instead of adding a door.

### Openings

**C9 Doors and windows sit in a wall (wall at both ends)**: RE-ADD, absolute
- `V = (D + N) · (1 − max( flank_h(W, L), flank_v(W, L) ))`, with L = Lmax_door for D and Lmax_win for N (computed per class, then summed).

**C10 Openings cut fully through the wall**: NEW, GT-hinged (Stage 1 found 28% of real plans fail an opening thickness/alignment check)
- `V = (D + N) · ( flank_h(W, L)·sum_v(W) + flank_v(W, L)·sum_h(W) )`.
- If the wall runs horizontally (walls left and right), there must be no wall directly above or below the opening, and the same for vertical walls. This catches "notch" openings that leave wall on one side.

**C11 A door joins two walkable spaces**: NEW (was validity rule 4), absolute
- `PASS' = S + OD`, `k = T_wall + 2`.
- `V = D · (1 − max( shift(PASS',0,+k)·shift(PASS',0,−k), shift(PASS',+k,0)·shift(PASS',−k,0) ))`.

**C12 A window has inside on one side and outside on the other**: NEW (was validity rule 5), absolute
- `V = N · (1 − max over the two axes of [ shift(S,+k)·shift(OUT,−k) + shift(OUT,+k)·shift(S,−k) ])`.
- Windows onto balconies count, because OUT includes outdoor.

**C13 Opening length in range**: NEW, GT-hinged
- For X = D (door limits) and X = N (window limits):
  - too short: `X − max( opn_h(X, Lmin), opn_v(X, Lmin) )`;
  - too long: `max( ero_h(X, Lmax+1), ero_v(X, Lmax+1) )`.

**C14 Door clearance (nothing blocks the door)**: NEW, absolute (G2 may switch it to GT-hinged; real bathrooms are tight)
- `V = dil(D, k_clear) · F`.

### Fixtures

**C15 Fixture is in a compatible room**: NEW, absolute
- `V = Σ_f P_f · (1 − dil( Σ_{c ∈ A(f)} P_c, k_fix ))`.
- Take A(f) from CubiCasa co-occurrence: the room types holding ≥ 5% of that fixture's pixels. Expected: toilet, bath → bathroom; sink → bathroom, kitchen; appliance → kitchen. Write A(f) to the config.

**C16 Fixtures stand against a wall**: NEW, GT-hinged (kitchen islands exist)
- `V = relu( F − rec(F · dil(W, 3), F, 32) )`. Every fixture blob must touch a wall.

**C17 Required fixtures (bathroom needs a toilet or bath; kitchen needs a sink or appliance)**: NEW, gated, training only
- For each GT bathroom or kitchen instance that lies **fully inside the window** and overlaps the hole: `L = 1 − smax(P_required, instance)`. Instances cut by the window border are skipped.
- Inference: a discrete rule on the parsed plan's room instances.

### Exterior

**C18 Railings border an outdoor space**: NEW, absolute
- `V = RL · (1 − dil(OD, 3))`.

**C19 Building envelope closed**: covered by C6 (interior touching outdoor or background directly).

### Check-only rules (verifier and detection, not losses)
These need whole-room areas or counts, which are noisy on 128-px training windows.
- **R1** room area range per type (CubiCasa percentiles)
- **R2** room count unchanged outside the hole
- **R3** plausible adjacency (e.g. bathroom not entered from the kitchen): table from CubiCasa statistics
- **R4** at least one exterior door per plan

---

## 2. Training integration

- `L = L_rec + g(t) · ramp(step) · Σ_i w_i · L_i`
- **Active from step 0** of the 16-class training. `ramp` goes linearly from 0 to 1 over the first 1,000 steps, to protect the warm-started weights.
- **Timestep gate (diffusion only):** g(t) = 1 for t ≤ 400, falling linearly to 0 at t = 800. At high t, x̂₀ is a blurry average and constraints there fight the reconstruction.
- **U-Net:** the same losses on softmax(logits), with no gate.
- **Warm start (class expansion):**
  - Initialise the 16-class models from the Stage 1 best checkpoints (constraint fine-tune diffusion; matched U-Net).
  - Input conv: copy the weights of classes that exist (wall, door, window, outdoor, background). Initialise every room-type and fixture channel from the old "room" channel. Do this for both the noisy-target and condition halves of the input.
  - Output conv: same mapping. Initialise railing from wall.
  - Result: the new model starts out equal to the Stage 1 model, with room types still undecided.

## 3. Weights: calibrate by gradient size, not loss value (fixes the Stage 1 bug)

1. Load the warm-started model and run 50 batches.
2. For each L_i, record `G_i = median ‖∂L_i/∂x̂₀‖`. This is the gradient with respect to the model output, so it is cheap (no parameter backprop). Record `G_0` for L_rec the same way.
3. Set `w_i = ρ · G_0 / G_i` with ρ = 0.05. If n·ρ > 0.4 (n = number of active constraints), use ρ = 0.4 / n.
4. Freeze the weights and write them to `configs/train16_constrained.yaml`. Log the G_i values too, for the report.

## 4. Gates (all must pass before tonight's launch)

- **G1 toy tests.** For each constraint, `V` is ≈ 0 on a small hand-made valid plan and > 0 on a matching broken plan (one broken toy per constraint). Save `outputs/constraint_toys.png`: each broken toy next to its violation map.
- **G2 real-plan check.** Compute mean_M V(Y) on 1,000 ground-truth training windows and write `outputs/constraint_gt_violations.md` (share of windows violating, mean value).
  - An absolute-mode constraint violated in > 10% of windows switches to GT-hinged.
  - Violated in > 40% of windows: it is not a rule in this data. Drop it, but **ask Venky first**.
- **G3 overfit test** (about 20 min, small model, 64 windows, all constraints on vs off). Reconstruction accuracy with constraints must be within about 2 points of without. Otherwise halve ρ for the loss with the largest gradient share and retest. (This is the test that caught the 100× bug in Stage 1.)
- **G4 speed.** Step time with all constraints must be ≤ 1.5× the step time without. Otherwise reduce `rec` iterations or add checkpointing.
- **Safety valve.** Any constraint that hasn't passed G1–G3 by about 10 pm stays out of training but is still used at inference (§5). Launch with the ones that passed.

## 5. Inference

**Guidance in x̂₀-space.** At each DDIM step with t ≤ 600:
- `x̂₀ ← x̂₀ − η · ∇_{x̂₀} Σ_i w_i L_i` (absolute mode, on M);
- then the usual DDIM update using the corrected x̂₀, with the known-region reset.

No backprop through the network, so it is cheap. Tune η on 200 validation windows (pick from {0, 0.1, 0.3, 1, 3}).

**Verifier (best-of-N).**
- Draw N samples per hole (N = 8 for evaluation, 4 for the live demo).
- Score each sample by its number of thresholded violations (C1–C18 absolute + R1–R4). Keep the lowest.
- Break ties by highest agreement (tolerant IoU) with the U-Net sketch.

**Hybrid sampling** as in Stage 1: U-Net sketch → re-noise to t₀ → DDIM with guidance → verifier.

## 6. The "both" missing-component detector

**Input**
- The 16-class one-hot plan, plus
- the absolute-mode V maps of every constraint that doesn't need a hole (everything except C2 and C5), plus R-rule maps where they have a pixel form.

**Output:** a per-pixel class for what is missing: none, wall, door, window, railing, room, toilet, sink, bath/shower, appliance.

**Backbone:** the same UNet2DModel family (no timestep) or an smp U-Net. Train on crops; run fully convolutionally on the whole plan at inference.

**Training data:** CubiCasa GT plans with simulated drafting omissions, drawn on the fly:

| Omitted | What the plan looks like afterwards |
|---|---|
| door | closed with wall (50%) or left as a plain gap of the neighbouring space (50%) |
| window | wall |
| wall segment | the neighbouring space extends through it |
| railing | outdoor / background |
| fixture | the surrounding room type |
| room | its interior walls removed and its type merged into a neighbour |

- Target = the omitted region, labelled with its type.
- 20% of training plans are left intact (all "none"), to teach the model not to raise false alarms.
- **Loss:** class-weighted cross-entropy + Dice on the missing classes.

**Final proposals** = union of rule proposals and detector proposals. Where they overlap, the type comes from the detector. Report rules-only, detector-only and both.

## 7. Evaluation hooks

- **Per-constraint table:** violation rate of every C_i and R_i for:
  - ground truth,
  - control model,
  - constrained model,
  - constrained + guidance,
  - constrained + guidance + verifier.
  
  This shows which constraint each part of the system fixes, without training 19 separate models.
- Stage 1 metrics on 16 classes: tolerant mIoU, component P/R/F1, validity.
- Per-component F1 (Hosseini & Furukawa style): correct type plus IoU ≥ 0.5 for openings and fixtures, ≥ 0.7 for rooms.
- Detection: precision and recall per missing type; false alarms per intact plan.

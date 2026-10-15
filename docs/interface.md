# Interface design (Gradio app)

This document describes the interactive interface (`app/app.py`) for the report's
interface-methodology section: how input is turned into a model input, how the
region to repair is chosen, which methods are offered, and what is reported.

**Settings are frozen.** Every model and sampling setting in the app is the one
selected on the val benchmark before the app was built (`outputs/selection.md`,
Day 7). Nothing was tuned on what the app shows. The gallery uses only **val**
plans; the test set stays untouched for the final numbers.

## 1. Architecture

| part | file | job |
|---|---|---|
| page | `app/app.py` | layout and event handlers only |
| settings | `configs/app.yaml` | checkpoint paths, frozen sampling settings, input limits |
| input pipeline | `src/interface/plan_io.py` | colours → classes, framing, file loading, validation |
| models | `src/interface/inference.py` | runs the 5 methods; loads each checkpoint on first use and keeps it |
| suggestions | `src/interface/suggest.py` | proposes masks around rule violations |
| metrics | `src/interface/report.py` | validity rules, and scores against the true plan |
| pictures | `src/interface/render.py` | plan ↔ screen images, mask layer, difference overlay |
| gallery | `src/interface/make_gallery.py` | 24 val cases, saved to `data/app/val_gallery.npz` |
| timing | `src/interface/time_methods.py` | one repair per method and hole size → `outputs/interface/cpu_timing.md` |

The app runs on the CPU and uses a GPU if one is available. If a checkpoint named in
`configs/app.yaml` is missing, start-up stops with a message naming the missing file
and where to get it.

**Lazy loading.** At start-up the app only checks that both checkpoint files exist. A network is
read from disk the first time a method needs it, and then kept for later repairs:
- nearest fill needs no network;
- the U-Net needs only the U-Net;
- refine-600 needs only the diffusion model;
- the hybrids need both.

There are only two networks, so at most two models are ever in memory. While a network loads, the
progress bar says "loading … (first use only)"; this takes under 1 s per network on the laptop. With both
models loaded, the app process uses about 0.75 GB of RAM.

The page has three steps: **1 plan → 2 region to repair → 3 repair**.

## 2. Input pipeline (step 1)

Every input ends up as our representation: a 128 × 128 label map with one of 6 classes per pixel.
Pixels whose content is unknown get the value 255; they are always repaired.

**(a) Samples.** 24 val-benchmark cases, 8 per hole size (small / medium / large).
Within each size, the cases cycle through the corruption types that occur at that size,
picked at random with a fixed seed. A sample comes with its benchmark mask, which is
pre-filled in step 2, and with its true plan, so accuracy can be reported.

**(b) Editor.** The plan is shown at 4× (512 px). The user paints with brushes in our 6 class colours.
To *delete* a component, paint over it with room, outdoor or background colour. Editing can
start from the current plan or from a blank (all-background) canvas. When the edit is used, the
screen picture becomes a label map in two steps:
1. every screen pixel snaps to the nearest class colour (Euclidean RGB distance);
2. each 4 × 4 block takes its majority class, which removes the blended colours at brush edges.

Pixels within 12 RGB units of the grey "unknown" colour become unknown. An edited plan has no
true plan, so no accuracy is reported. It must contain at least 100 wall pixels.

**(c) Upload.**
- **PNG (or JPEG/BMP/WebP) in our class colours.**
  - Each pixel snaps to the nearest class colour; transparent pixels become background, and grey becomes unknown.
  - The image is refused if more than 10% of its pixels are farther than 60 RGB units from every class colour, i.e. it is not drawn in our palette.
  - It is then **framed exactly like the data pipeline**: the bounding box of walls + doors + windows is scaled so its longer side is 120 px, centred in 128 × 128, with nearest-neighbour sampling. Parts outside the frame are cropped; empty space is background.
  - A 128 × 128 input whose structure already spans 112–124 px is kept as is, because re-sampling a plan that is already framed would distort its 2-px walls.
- **.npy label map**: a 2-D array of class ids 0–5 (255 = unknown), or a (6, H, W) one-hot/score array, which is reduced with argmax. It is framed the same way and loaded with `allow_pickle=False`, so a user file can never run code.
- **Validation** — each failure gives a message saying what is wrong and what is expected:
  - unreadable file;
  - unsupported type;
  - side under 16 or over 4,096 px;
  - colours not from our palette;
  - wrong array shape or class ids;
  - no walls, doors or windows to frame on;
  - fewer than 100 wall pixels after framing.

## 3. Region to repair (step 2)

The user brushes the region to repair on the plan; it is shown in our grey. A 128-px pixel is masked
if at least 25% of its 4 × 4 screen block is brushed. Unknown pixels are always added to the mask.

**Mask shape vs training.** The models were trained and evaluated on rectangular masks: a component's
bounding box plus 2–6 px of padding, or a random rectangle. Brushed masks can have any shape,
which is outside the training distribution. The benchmark masks of the samples are rectangles.

**Suggest suspicious areas (assistive).** The button runs four of our architectural rules
(`src/eval/arch_metrics.py`) on the whole plan:
- unreachable rooms;
- dead-end walls;
- floating wall fragments;
- doors not between two spaces.

It proposes a mask made of a 4-px-padded box around every violation. The proposal goes into the
mask editor, where the user can accept or edit it. The page labels it as assistive and not
exhaustive, for two reasons:
- **Real plans also break these rules.** On 500 real val plans: dead ends 11.0%, door spaces 1.6%, unreachable rooms 1.2%, fragments 0.0%.
- **Many kinds of damage break none of them**, e.g. a missing window, or a wall that moved.

## 4. Methods (step 3)

| method (dropdown) | what it does | val result (`outputs/selection.md`) |
|---|---|---|
| **Size-adaptive hybrid** (default) | matched U-Net draft → one-hot → noised to t₀ → constraint fine-tune finishes it (stochastic DDIM, eta 1, known-region reset); t₀ = 400 / 600 / 700 for small / medium / large holes | tol. mIoU 0.842, validity 88.0% |
| Hybrid, t₀ = 600 | the same with t₀ = 600 for every hole | 0.842, 87.4% |
| Matched U-Net | one forward pass | 0.846, 80.4% |
| Diffusion, refine-600 | constraint fine-tune: its own one-shot guess, refined from t₀ = 600 | 0.801, 83.6% |
| Nearest fill | baseline: each hole pixel copies the nearest known pixel | 0.489, 36.0% |

**Hole size at inference.** In the benchmark, the size bin is the mask area as a percentage of the
building area (wall + door + window + room) of the *clean* plan, which the app does not have. The
app measures the building on the plan completed by the U-Net draft, which the hybrid computes anyway.
On all 7,772 val cases this gives the benchmark's size bin in 7,771 cases (99.99%). Measuring the
known building pixels plus the mask instead gives 99.55%. This choice was made on val before any
app use and changes no model or sampling setting.

**Seed.** The diffusion methods are stochastic. The seed field fixes the random numbers, so the same
seed gives the same repair. **Generate another** adds 1 to the seed and repairs again, showing a
different plausible layout. The U-Net and nearest fill are deterministic.

Pixels outside the mask are never changed. The progress bar follows the denoising steps.

## 5. Outputs and metrics

Three pictures:
- **input**, with the region to repair in grey;
- **repaired plan**;
- **difference overlay**: the repaired plan faded, with what changed inside the mask highlighted:
  - magenta: wall/door/window pixels the repair drew;
  - cyan: wall/door/window pixels that were there before and are gone. This only applies where the input was known, e.g. when the user masks an existing part to redraw it;
  - amber: other class changes where the input was known.

**Architectural rules.** The 6 validity rules from the evaluation are counted inside the repaired
region (the mask grown by 3 px, as in `src/arch_eval.py`):
1. no dead-end walls;
2. no floating wall fragments;
3. doors and windows bridge two wall ends;
4. doors connect two spaces;
5. windows face outside;
6. every room is reachable through doors.

How a rule passes:
- **with a true plan (samples):** the repair has no more violations than the true plan in the same region. This is exactly the evaluation's validity definition.
- **without a true plan:** only 0 violations count as a pass. The page notes that real plans also break some rules.

**Against the true plan (samples only):** tolerant mIoU and door/window/room component F1 inside the
mask, computed with the evaluation code (`src/eval/metrics.py`). "n/a" means the class is absent
from both the repair and the truth inside the mask.

**Runtime** of the repair is shown under the method name.

## 6. Speed

One repair per method and hole size, on the development laptop:
- cases: the first small, medium and large gallery case, seed 0, after a warm-up;
- machine: Intel CPU, 10 threads, PyTorch 2.14 (CPU), fp32;
- made with `python -m src.interface.time_methods` → `outputs/interface/cpu_timing.md`.

| method | small hole | medium hole | large hole |
|---|---|---|---|
| **Size-adaptive hybrid** (t₀ 400 / 600 / 700) | 29.3 s | 44.2 s | 52.3 s |
| Hybrid, t₀ = 600 | 45.4 s | 45.8 s | 46.7 s |
| Matched U-Net | 1.4 s | 1.4 s | 1.4 s |
| Diffusion, refine-600 | 45.1 s | 45.4 s | 45.5 s |
| Nearest fill | < 0.01 s | < 0.01 s | < 0.01 s |

Loading a checkpoint (once, on first use): 0.7 s (U-Net), 0.6 s (diffusion).

**Why the times differ:**
- Each denoising step is one network pass, about 1.4 s on this CPU.
- The diffusion methods run only the steps of the 50-step DDIM schedule at or below t₀: 21 steps for
  t₀ = 400, 31 for 600 and 36 for 700. Time therefore grows with t₀, and the size-adaptive hybrid is
  fastest on small holes.
- A U-Net repair is a single pass.

No speed-up that would change the results was used (`docs/decisions.md`).

## 7. Testing

**Unit tests.** `tests/test_interface.py` runs as part of `python -m pytest tests/` and needs no
models or data. It covers:
- framing and colour snapping;
- PNG and `.npy` round trips;
- 7 kinds of malformed input, each refused with `InputError`;
- brush layer → mask, and the suggestions;
- size bins, the rule table and the pictures;
- the missing-checkpoint message.

**Functional test of the app** (2026-10-14). To save memory, the app's event handlers in
`app/app.py` were called directly from Python with the real checkpoints, instead of clicking
through a browser. 37 of 37 checks passed:
- **Samples:** picking a gallery case fills step 2 with exactly its benchmark mask.
- **All 5 methods** repair it.
  - Pixels outside the mask never change.
  - The same seed gives the same repair.
  - *Generate another* raises the seed by 1.
  - For a deterministic method, a note says that a new seed changes nothing.
- **Size-adaptive t₀:** a small-hole case got t₀ = 400 and a large one t₀ = 700.
- **Lazy loading:** there were never more than 2 models in memory.
  - Nearest fill loaded no model.
  - The U-Net loaded only the U-Net.
  - The first hybrid added the diffusion model.
- **Editor:**
  - A wall drawn on the canvas is read back exactly at 128 × 128, and the rest of the plan is unchanged.
  - The edited plan can be brushed and repaired. With no true plan, only the rules are reported.
- **Uploads:**
  - 2 valid files (a PNG and a `.npy`) are accepted.
  - A blank PNG, a photo-like PNG and a text file renamed `.npy` are each refused with a message.
- **Errors:** each step gives a clear message when it is used out of order (no plan, empty mask,
  empty editor, no file).

**Browser check.** The page was opened in Chrome. A large-hole sample was repaired with the
size-adaptive hybrid (t₀ = 700, 50 s, 6/6 rules pass). The screenshots are in `outputs/interface/`.

## 8. Limitations

- **Speed on a CPU.** A diffusion repair takes about 30–52 s on a laptop CPU (section 6), against 0.2–0.4 s on a T4 GPU.
- **The mask editor shows the plan slightly blurred**, because the image editor rescales its canvas. The plan behind it is unchanged.
- **Free-form masks** are outside the training distribution, which used rectangles (section 3).
- **Uploads must use our class colours and drawing conventions** (2-px walls, doors and windows drawn in the wall line). A photo or a scanned architectural drawing is refused, not converted.
- **Suggestions cover only 4 rules** and also flag legitimate features (open passages next to free-standing wall ends).
- **The validity rules are heuristics**, validated on real plans, with the rates in `docs/decisions.md`. Without a true plan, "fail" means "worth a look", not "wrong".

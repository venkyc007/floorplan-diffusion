# Code walkthrough video — 8-minute script

What to open, which line to jump to, and what to say, in order. It follows the five allotted deliverables, then the
models and the interface. About 950 spoken words: at a normal pace that is ~7 minutes of talking, plus ~1 minute for
switching tabs. Line numbers are for the final code (`main`, 2026-10-25); if the code changes, check them again.

**Before recording**
- Open the project folder in VS Code; close everything else. Font size 16+, so the code is readable on video.
- Pre-open these tabs **in this order** (left to right). The script moves through them one by one:
  1. `README.md` 2. `src/stage1/data/build_dataset.py` 3. `src/stage1/data/corruptions.py`
  4. `src/stage2/detect/image_eval.py` 5. `src/stage1/models/diffusion.py` 6. `src/stage1/models/unet_baseline.py`
  7. `src/stage1/interface/inference.py` 8. `src/stage1/models/constraints.py`
  9. `configs/stage1/finetune_constraints.yaml` 10. `src/stage2/constraints/constraints.py`
  11. `src/stage2/complete/models16.py` 12. `src/stage2/parse/losses.py` 13. `src/stage1/train.py`
  14. `configs/stage2/train16_constrained.yaml` 15. `src/stage2/pipeline/run.py` 16. `src/stage1/eval/metrics.py`
  17. `src/stage2/pipeline/e2e_val.py` 18. `outputs/stage1/stage1_test.md` 19. `outputs/stage2/final_test.md`
  20. `outputs/stage2/test_components.md` 21. `demo_app/app.py`
- Jump with **Ctrl+G** + the line number. Click the line so it is highlighted, talk, move on; don't scroll around.
- Keep the camera on (faces must be visible).

## The script

### 1 · Intro (0:00–0:35)

**Open** `README.md` → **Ctrl+G 12** (the Stage 1 / Stage 2 table).

> "Hi, I'm Venkatesh, team One Hot Mencoding. My project is diffusion-based reconstruction of missing floor plan
> components: an engineer forgets a door or a wall while drawing, and the model puts it back. I did it in two
> stages. In Stage 1, on clean ResPlan maps, the missing region is given and the model redraws it. In Stage 2, on
> real CubiCasa5K drawings, nothing is given: the system must find what is missing and then draw it. I'll follow
> the five deliverables."

### 2 · Deliverable 1: dataset and split (0:35–1:10)

**Open** `src/stage1/data/build_dataset.py` → **Ctrl+G 125**, then point at **131**, **149** and **174**.

> "Stage 1 uses ResPlan: about seventeen thousand plans, turned into 128-by-128 maps with six classes. Line 125
> leaves out ResPlan's augmented copies. Line 131 finds near-duplicate plans and removes them. Line 149 splits the
> rest 80/10/10 in groups of similar plans, so similar plans never land in both train and test, and line 174
> checks exactly that. Stage 2 uses CubiCasa5K's official split."

### 3 · Deliverable 1: controlled missing components (1:10–1:50)

**Open** `src/stage1/data/corruptions.py` → **Ctrl+G 7** (the list of types), then **Ctrl+G 59**.

> "This file removes parts of a plan in a controlled way. Five types: a door, a window, a wall segment, a whole
> room, or a random region. Every case also records the location, interior or exterior, the position, centre or
> edge, and the size: small, medium or large. One detail, at line 59: the mask is never the exact shape of what was
> removed. It is the bounding box grown by a random 2 to 6 pixels, so the mask doesn't give the answer away."

### 4 · Deliverable 1: omissions on real drawings (1:50–2:10)

**Open** `src/stage2/detect/image_eval.py` → **Ctrl+G 92** (`omit`).

> "For Stage 2, parts are erased from the drawing image itself: a door, a window, a wall, a railing, a fixture, a
> room's walls, a whole room or a region, with the same metadata as Stage 1."

### 5 · Deliverable 2: the diffusion model (2:10–2:55)

**Open** `src/stage1/models/diffusion.py` → **Ctrl+G 43**, point at **53**, then **Ctrl+G 93**.

> "Deliverable two is the diffusion model. It is Palette-style conditional diffusion: the network sees the noisy
> plan, the corrupted plan and the mask, and predicts the clean plan directly. The loss is a mean squared error, and
> line 53 weights the pixels inside the hole five times more. Sampling uses DDIM, and at line 93, after every step,
> everything outside the mask is reset to the known plan. That is how the surrounding structure is preserved: only
> the hole changes."

### 6 · Deliverable 2: the hybrid (2:55–3:25)

**Open** `src/stage1/models/unet_baseline.py` → **Ctrl+G 28**. Then **open** `src/stage1/interface/inference.py` →
**Ctrl+G 143**, point at **153**.

> "A plain U-Net, trained with cross-entropy, was more accurate than diffusion alone, so the final method is a
> hybrid. At line 143 the U-Net draws a sketch. At line 153 the sketch is noised to a level t0, 400, 600 or 700
> depending on the hole size, and the diffusion model finishes it."

### 7 · Deliverable 3: structure-aware losses, Stage 1 (3:25–4:05)

**Open** `src/stage1/models/constraints.py` → **Ctrl+G 82**, then **107**, then **148**. Then **open**
`configs/stage1/finetune_constraints.yaml` → **Ctrl+G 27**.

> "Deliverable three is structure. cldice_loss compares soft skeletons of the predicted and true walls, so broken
> walls cost something: that is wall connectivity. closure_loss penalises rooms that leak to the outside. At line
> 148, rule losses only count where the prediction is worse than the real plan, so the model can't win by deleting
> doors. In the config, the weights are 0.01 and 0.03. My first weights were about 100 times too strong, so I
> calibrated them by gradient size."

### 8 · Deliverable 3: the Stage 2 rule library and losses (4:05–4:50)

**Open** `src/stage2/constraints/constraints.py` → **Ctrl+G 390** (`REGISTRY`). Then **open**
`src/stage2/complete/models16.py` → **Ctrl+G 86**. Then **open** `src/stage2/parse/losses.py` → **Ctrl+G 44**.

> "Stage 2 has a library of 18 rules, C1 to C18: walls connect, walls continue straight into the hole, openings sit
> in a wall, a door joins two spaces, and so on. Each rule gives a violation map, used four ways: as a training
> loss, as detector input, in the verifier, and as guidance. In the 16-class diffusion, line 86 adds 14 of these
> terms to the loss. The parser, line 44, uses weighted cross-entropy plus Dice, so thin doors count as much as
> background."

### 9 · Training: optimizer, learning rate, schedule (4:50–5:25)

**Open** `src/stage1/train.py` → **Ctrl+G 261**, then **74**, then **126**. Then **open**
`configs/stage2/train16_constrained.yaml` → **Ctrl+G 33**.

> "Training uses AdamW with weight decay 0.01. The learning rate warms up linearly, then decays to zero along a
> cosine. Each step runs in half precision, clips the gradients at 1.0 and updates an averaged copy of the weights.
> Learning rates are 2e-4 for the U-Net, 1e-4 for diffusion and 2e-5 for fine-tunes, like this config. Everything
> trained on Kaggle, on two T4 GPUs. All settings were chosen on the validation set; the test sets were used only for the final evaluation."

### 10 · The full Stage 2 method (5:25–5:55)

**Open** `src/stage2/pipeline/run.py` → **Ctrl+G 7** (the list of steps), then **Ctrl+G 498**, point at **503** and
**536**.

> "This is the full Stage 2 method. PARSE, line 498: a ResNet-34 U-Net turns the drawing into a 16-class map. CHECK,
> line 503: rules and learned detectors flag what is missing, with its type. FILL, line 536: the 16-class hybrid
> fills each hole. VERIFY: it draws several fills and keeps the one with the fewest rule violations. Passing boxes
> instead is the manual mode."

### 11 · Deliverable 4: component-level evaluation (5:55–6:35)

**Open** `src/stage1/eval/metrics.py` → **Ctrl+G 81**, then **109**. Then **open** `src/stage2/pipeline/e2e_val.py`
→ **Ctrl+G 161**.

> "Deliverable four is component-level evaluation. match_components pairs predicted and true doors, windows and
> rooms one to one with the Hungarian algorithm; a match needs an IoU of 0.5. That gives precision, recall and F1
> per component. The tolerant IoU allows one pixel of slack, because walls are thin. In Stage 2, line 161 records
> for every omission: was it found, was the type right, was it fixed, and how many false additions."

### 12 · Deliverable 5: robustness and ablation (6:35–7:30)

**Open** `outputs/stage1/stage1_test.md` → **Ctrl+G 62**, point at the headings at **72** and **122**. Then **open**
`outputs/stage2/final_test.md` → **Ctrl+G 28** (the "all" rows), then **Ctrl+G 40** (the ablation "all" row). Then
**open** `outputs/stage2/test_components.md` → **Ctrl+G 12**.

> "Deliverable five. On the Stage 1 test set, the hybrid gets tolerant mIoU 0.846 and 88% validity, against 81% for
> the U-Net alone, and the results are broken down by type and by hole size. On the Stage 2 test, the U-Net sketch
> alone makes 18% valid fills, and the diffusion finish raises that to 52%. The ablation compares diffusion with and
> without the constraint losses: no measurable difference, the confidence interval contains zero. Per omission
> kind, room walls and windows work best, fixtures and railings worst."

### 13 · Models, interface, close (7:30–8:00)

**Open** `README.md` → **Ctrl+G 137** (Trained models). Then **open** `demo_app/app.py` → **Ctrl+G 229**, then **415**.

> "The trained models are in the Kaggle dataset floorplan-stage2-final. The interface is one Gradio app, with a
> Stage 1 tab and a Stage 2 tab where you can upload any drawing and see every step. The demo video shows it running.
> Every choice and its reason is logged in docs/decisions.md. Thank you."

## Timing tips

- Rehearse once with a stopwatch. If you run long, cut part 4 to one sentence and skip the parser in part 8: both are
  also in the report.
- If you run short, add one sentence in part 12 on the large-hole models (`final_test.md` lines 25 and 27: tolerant mIoU
  49.8 → 55.4 on large holes).

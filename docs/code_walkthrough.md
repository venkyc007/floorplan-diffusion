# Code walkthrough video — 10-minute script

Exactly what to open, when, and what to point at. Organised around the five allotted deliverables, then the
submission items (models, interface, metrics). Line ranges are generated from the code
(`python -m src.doc_lines docs/code_walkthrough.md` rewrites every `L…` range), so they match this commit.

**Before recording**
- Open the project folder in VS Code; close everything else. Font size 16+, so the code is readable on video.
- Pre-open these tabs **in this order** (left to right) — the script moves through them one by one:
  1. `README.md` 2. `src/stage1/data/corruptions.py` 3. `src/stage2/detect/image_eval.py`
  4. `src/stage1/models/diffusion.py` 5. `src/stage1/models/unet_baseline.py` 6. `src/stage1/models/constraints.py`
  7. `src/stage2/parse/losses.py` 8. `src/stage2/complete/models16.py` 9. `src/stage2/constraints/constraints.py`
  10. `src/stage1/train.py` 11. `configs/stage2/train16_constrained.yaml` 12. `src/stage2/pipeline/run.py`
  13. `src/stage1/eval/metrics.py` 14. `src/stage2/pipeline/e2e_val.py` 15. `outputs/stage2/test_components.md`
  16. `outputs/stage1/stage1_test.md` 17. `outputs/stage2/final_test.md` 18. `demo_app/app.py`
- Use **Ctrl+G** (go to line) to jump to the lines below; keep camera on (faces must be visible).

## The script

| time | deliverable | open | go to | say (in your own words) |
|---|---|---|---|---|
| **0:00–0:40** | intro | `README.md` | the stage table at the top, then the "Deliverables checklist" | task (engineer misses a component → reconstruct it with diffusion); Stage 1 = mask given (ResPlan), Stage 2 = real drawings, find + fill (CubiCasa5K); repo layout `src/stage1`, `src/stage2`, `configs`, `outputs`, `docs` |
| **0:40–1:30** | **1. dataset** | (still README) → `src/stage1/data/build_dataset.py` | `dedup_and_split` <!--src/stage1/data/build_dataset.py::dedup_and_split-->L116–160, `check_no_leakage` <!--src/stage1/data/build_dataset.py::check_no_leakage-->L174–189 | ResPlan 17k vector plans → 128×128 maps with 6 classes; augmented copies excluded; near-duplicates removed; **group-wise 80/10/10 split** so similar plans never cross train/val/test; Stage 2: CubiCasa5K official split (`src/stage2/data/build_cubicasa.py` `read_split_lists` <!--src/stage2/data/build_cubicasa.py::read_split_lists-->L53–61) |
| **1:30–2:20** | **1. missing-component generation** | `src/stage1/data/corruptions.py` | module docstring (top), `padded_box_mask` <!--src/stage1/data/corruptions.py::padded_box_mask-->L59–68, `describe` <!--src/stage1/data/corruptions.py::describe-->L169–192 | removal **by type** (door, window, wall segment, room, region), **location** (interior/exterior), **position** (centre/edge), **size** (small/medium/large); the mask is the box + 2–6 px so it never reveals the shape |
| **2:20–2:50** | 1. (Stage 2) | `src/stage2/detect/image_eval.py` | `omit` <!--src/stage2/detect/image_eval.py::omit-->L92–131 | on real drawings the component is erased from the image itself: door drawn over / leaf erased, window, wall, railing, fixture, room walls, whole room, region; same metadata |
| **2:50–3:40** | **2. diffusion model** | `src/stage1/models/diffusion.py` | `DiffusionInpainter.loss` <!--src/stage1/models/diffusion.py::DiffusionInpainter.loss-->L43–65, `DiffusionInpainter.predict` <!--src/stage1/models/diffusion.py::DiffusionInpainter.predict-->L67–94 | Palette-style: input = noisy plan + corrupted plan + mask; predicts the clean plan x0; cosine schedule, 1,000 steps. **Loss 1: MSE on x0, hole pixels ×5.** Sampling: DDIM, known region reset every step → **surrounding structure preserved** |
| **3:40–4:05** | 2. the sketch model | `src/stage1/models/unet_baseline.py` | `UNetInpainter.loss` <!--src/stage1/models/unet_baseline.py::UNetInpainter.loss-->L28–33 | **Loss 2: cross-entropy, hole ×5.** Final method = hybrid: this U-Net draws a sketch, it is noised to t₀ (400/600/700 by hole size) and the diffusion model finishes it (`src/stage1/interface/inference.py` `Repairer.repair` <!--src/stage1/interface/inference.py::Repairer.repair-->L115–160) |
| **4:05–4:50** | **3. structure-aware constraints** | `src/stage1/models/constraints.py` | `cldice_loss` <!--src/stage1/models/constraints.py::cldice_loss-->L82–89, `closure_loss` <!--src/stage1/models/constraints.py::closure_loss-->L107–112, `constraint_losses` <!--src/stage1/models/constraints.py::constraint_losses-->L131–160 | **Loss 3: soft clDice** = wall connectivity via soft skeletons; **Loss 4: room closure** = rooms must not leak to the outside; both relative to the ground truth; weights 0.01 / 0.03 (first ~100× too strong → **calibrated by gradient size**) |
| **4:50–5:20** | 3. Stage 2 constraint library | `src/stage2/constraints/constraints.py` | `REGISTRY` (line 390), `constraint_loss` <!--src/stage2/constraints/constraints.py::constraint_loss-->L418–432 | C1–C18: one function per rule (connectivity, closure, openings in walls, **wall alignment C5**, door adjacency…) → a violation map used **four ways**: training loss, detector input, verifier, guidance |
| **5:20–5:55** | 2+3. Stage 2 losses | `src/stage2/parse/losses.py` → `src/stage2/complete/models16.py` | `ParserLoss` <!--src/stage2/parse/losses.py::ParserLoss-->L35–44, `dice_loss` <!--src/stage2/parse/losses.py::dice_loss-->L20–32; then `constraint_terms` <!--src/stage2/complete/models16.py::constraint_terms-->L41–52, `Diffusion16.loss` <!--src/stage2/complete/models16.py::Diffusion16.loss-->L68–87, `UNet16.loss` <!--src/stage2/complete/models16.py::UNet16.loss-->L122–136 | **Loss 5: parser & detectors = class-weighted cross-entropy + Dice** (Dice = overlap per class, so thin doors count as much as background). **Loss 6: 16-class fill = MSE on x0 (hole ×5) + 14 constraint terms**, switched on at low noise and ramped in |
| **5:55–6:35** | training: optimizer, LR, schedule | `src/stage1/train.py` → `configs/stage2/train16_constrained.yaml` | `lr_factor` <!--src/stage1/train.py::lr_factor-->L74–79, `train_step` <!--src/stage1/train.py::train_step-->L126–143, `EMA` <!--src/stage1/train.py::EMA-->L108–123; in the yaml: `train:` block and `constraints: weights` | **AdamW** (weight decay 0.01); **LR schedule = linear warm-up then cosine to 0**; LRs 2e-4 (U-Net), 1e-4 (diffusion), 2e-5 (fine-tunes); fp16, gradient clipping 1.0, EMA 0.999; checkpoint every 30 min + auto-resume; Kaggle T4 × 2. Calibrated weights listed in the yaml; `control` config identical without them |
| **6:35–7:15** | the full Stage 2 method | `src/stage2/pipeline/run.py` | module docstring, `detect_parts` <!--src/stage2/pipeline/run.py::detect_parts-->L154–197, `run_pipeline` <!--src/stage2/pipeline/run.py::run_pipeline-->L490–550 | PARSE (segmentation U-Net, ResNet-34) → CHECK (rules + detectors v2/v3) → FILL (16-class hybrid; B4 model for large holes) → VERIFY (best of N by fewest rule violations); manual mode = `boxes=` |
| **7:15–8:05** | **4. component-level evaluation** | `src/stage1/eval/metrics.py` → `src/stage2/pipeline/e2e_val.py` | `match_components` <!--src/stage1/eval/metrics.py::match_components-->L81–93, `tolerant_counts` <!--src/stage1/eval/metrics.py::tolerant_counts-->L109–124, `sample_stats` <!--src/stage1/eval/metrics.py::sample_stats-->L137–186; `score` <!--src/stage2/pipeline/e2e_val.py::score-->L136–171 | **component detection precision / recall / F1** (Hungarian matching, IoU ≥ 0.5) for doors, windows, rooms; tolerant IoU (1-px slack for thin walls); connectivity error, room leak; validity. Stage 2 per omission: **found, right type, fixed, false additions** |
| **8:05–9:10** | **5. robustness + ablation** | `outputs/stage1/stage1_test.md` → `outputs/stage2/final_test.md` → `outputs/stage2/test_components.md` | Stage 1: "Test, type=…" and "size_bin=…" sections + paired bootstrap; Stage 2 final test: FILL table + "Ablation: constrained vs control"; components test: "By omission kind", "By location", "By position" | **robustness** = results by type, location, position, size; **ablation** = U-Net alone vs + diffusion (18% → 52% valid fills), constrained vs control (no measurable difference, CI shown), B4 vs 128-px model on large holes. Component test (371 test drawings): 41% fixed; best room walls / windows, worst fixtures / railings |
| **9:10–9:40** | trained models + interface | `README.md` "Trained models" → `demo_app/app.py` | the checkpoint table; in app.py `stage1_tab` <!--demo_app/app.py::stage1_tab-->L229–314 and `stage2_tab` <!--demo_app/app.py::stage2_tab-->L415–483 | checkpoints in the Kaggle dataset `floorplan-stage2-final`; one Gradio app, Stage 1 + Stage 2 tabs, upload any drawing, shows every step and scores (shown in the demo video) |
| **9:40–10:00** | wrap-up | `docs/results.md` §4 | the weak-spots list | what worked (structure from the diffusion finish, the fill), what did not (detection of large omissions, false alarms); every decision logged in `docs/decisions.md` |

## The six loss functions in one place (for the 3:40–5:55 stretch)

| # | model | loss | where |
|---|---|---|---|
| 1 | Stage 1 diffusion | MSE on predicted x0, hole ×5 | `src/stage1/models/diffusion.py` `DiffusionInpainter.loss` <!--src/stage1/models/diffusion.py::DiffusionInpainter.loss-->L43–65 |
| 2 | Stage 1 U-Net (sketch) | cross-entropy, hole ×5 | `src/stage1/models/unet_baseline.py` `UNetInpainter.loss` <!--src/stage1/models/unet_baseline.py::UNetInpainter.loss-->L28–33 |
| 3 | Stage 1 fine-tune | + soft clDice (connectivity) | `src/stage1/models/constraints.py` `cldice_loss` <!--src/stage1/models/constraints.py::cldice_loss-->L82–89 |
| 4 | Stage 1 fine-tune | + room closure | `src/stage1/models/constraints.py` `closure_loss` <!--src/stage1/models/constraints.py::closure_loss-->L107–112 |
| 5 | parser, detectors v2 / v3 | class-weighted cross-entropy + Dice (parser v2B adds 8 constraint terms, `src/stage2/parse/train_v2.py` `constraint_terms` <!--src/stage2/parse/train_v2.py::constraint_terms-->L105–119) | `src/stage2/parse/losses.py` `ParserLoss` <!--src/stage2/parse/losses.py::ParserLoss-->L35–44 |
| 6 | Stage 2 16-class U-Net / diffusion / B4 | CE or MSE (hole ×5) + 14 calibrated constraint terms | `src/stage2/complete/models16.py` `constraint_terms` <!--src/stage2/complete/models16.py::constraint_terms-->L41–52 |

Gradient-size weight calibration: `src/stage2/complete/calibrate.py` `weights_from` <!--src/stage2/complete/calibrate.py::weights_from-->L75–97.

## Timing tips
- Rehearse once with a stopwatch; the table adds up to exactly 10:00. If you run long, shorten 6:35–7:15 (the
  pipeline is also shown in the demo video).
- Don't scroll searching: Ctrl+G to the line, highlight 3–5 lines, talk, move on.

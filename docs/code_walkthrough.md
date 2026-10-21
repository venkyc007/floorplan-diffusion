# Code walkthrough map

Where each topic of the IE643 code-walkthrough video lives in the code, with a 10-minute plan. Line numbers are
generated from the code (`python -m src.doc_lines docs/code_walkthrough.md` rewrites every `L…` range), so they match
the files in this commit.

## 10-minute plan

| time | topic | show |
|---|---|---|
| 0:00–0:40 | the repo and the two stages | `README.md` (table + map), folder tree, `docs/decisions.md` (final configuration box at the top) |
| 0:40–1:40 | Stage 1 data: collection, preprocessing, split | §1 below |
| 1:40–2:20 | controlled corruption + fixed benchmarks | §2 |
| 2:20–3:20 | networks and loss functions | §3 |
| 3:20–4:10 | training: optimizer, learning rate, schedule, EMA, checkpoints | §4 |
| 4:10–4:50 | metrics, validation, model selection, the one test run | §5 |
| 4:50–5:40 | Stage 2 data: CubiCasa5K, 16 classes, erasure | §6 |
| 5:40–6:30 | PARSE | §7 |
| 6:30–7:30 | CHECK: constraint library, rules, detectors, tuning | §8 |
| 7:30–8:30 | FILL + VERIFY: 16-class models, warm start, hybrid sampler, best-of-N | §9 |
| 8:30–9:30 | the end-to-end pipeline, the final test, the app | §10 |
| 9:30–10:00 | wrap-up: what worked, what did not (`docs/results.md` §4) | |

## 1. Stage 1 data — collection, preprocessing, split

| what | where |
|---|---|
| load ResPlan safely (allow-listed unpickler) | `src/stage1/data/resplan.py` `load_plans` <!--src/stage1/data/resplan.py::load_plans-->L67–70 |
| vector plan → 128 × 128 6-class map (framing on walls + doors + windows, paint order) | `src/stage1/data/rasterize.py` `make_transform` <!--src/stage1/data/rasterize.py::make_transform-->L37–58, `rasterize_plan` <!--src/stage1/data/rasterize.py::rasterize_plan-->L71–91 |
| channel order (one place) | `src/stage1/data/classes.py` |
| near-duplicate removal (cosine over 8 rotations/flips, then tolerant IoU ≥ 0.90) | `src/stage1/data/dedup.py` `find_candidates` <!--src/stage1/data/dedup.py::find_candidates-->L64–86, `tolerant_iou` <!--src/stage1/data/dedup.py::tolerant_iou-->L91–113, `remove_duplicates` <!--src/stage1/data/dedup.py::remove_duplicates-->L133–159 |
| **train / val / test split** (similarity groups never cross splits) | `src/stage1/data/split.py` `split_groups` <!--src/stage1/data/split.py::split_groups-->L12–42; leakage assertions `src/stage1/data/build_dataset.py` `check_no_leakage` <!--src/stage1/data/build_dataset.py::check_no_leakage-->L174–189 |
| settings | `configs/stage1/data.yaml` |

## 2. Controlled corruption + benchmarks

| what | where |
|---|---|
| mask rule: bounding box + random 2–6 px per side (never the exact shape) | `src/stage1/data/corruptions.py` `padded_box_mask` <!--src/stage1/data/corruptions.py::padded_box_mask-->L59–68 |
| remove one part by type (door, window, wall segment, room, region) + metadata (location, position, size bin) | `pick_part` <!--src/stage1/data/corruptions.py::pick_part-->L142–164, `describe` <!--src/stage1/data/corruptions.py::describe-->L169–192, `corrupt` <!--src/stage1/data/corruptions.py::corrupt-->L204–215 |
| 7-channel model input (6 one-hot with the hole zeroed + the mask) | `to_model_input` <!--src/stage1/data/corruptions.py::to_model_input-->L238–242 |
| training data: a fresh random corruption + random rotation/flip per sample | `src/stage1/data/dataset.py` `CorruptionDataset` <!--src/stage1/data/dataset.py::CorruptionDataset-->L51–105 |
| fixed seeded val / test benchmarks with checksums | `src/stage1/data/make_benchmark.py` `make_benchmark` <!--src/stage1/data/make_benchmark.py::make_benchmark-->L43–65 |

## 3. Networks and loss functions

| what | where |
|---|---|
| shared backbone: diffusers UNet2DModel, channels 64/128/256/256, attention at 16 × 16 (25.3M parameters) | `src/stage1/models/backbone.py` `build_unet` <!--src/stage1/models/backbone.py::build_unet-->L12–33 |
| U-Net inpainter: 7 channels in → 6 logits; **loss = cross-entropy, hole pixels × 5** | `src/stage1/models/unet_baseline.py` `UNetInpainter.loss` <!--src/stage1/models/unet_baseline.py::UNetInpainter.loss-->L28–33 |
| diffusion inpainter (Palette-style): 13 channels in (noisy one-hot + condition), predicts x0, cosine schedule, 1,000 steps; **loss = MSE on x0, hole × 5, + constraint losses** | `src/stage1/models/diffusion.py` `DiffusionInpainter.loss` <!--src/stage1/models/diffusion.py::DiffusionInpainter.loss-->L43–65 |
| DDIM sampling with the known region overwritten at each step | `DiffusionInpainter.predict` <!--src/stage1/models/diffusion.py::DiffusionInpainter.predict-->L67–94 |
| structural losses: soft clDice (wall connectivity), room closure; ground-truth-relative | `src/stage1/models/constraints.py` `cldice_loss` <!--src/stage1/models/constraints.py::cldice_loss-->L82–89, `closure_loss` <!--src/stage1/models/constraints.py::closure_loss-->L107–112 |
| the final hybrid: U-Net sketch → noised to t₀ → stochastic DDIM | `src/stage1/sampling_experiment.py` `refine` <!--src/stage1/sampling_experiment.py::refine-->L55–78; size-adaptive t₀ in `src/stage1/interface/inference.py` `Repairer.repair` <!--src/stage1/interface/inference.py::Repairer.repair-->L115–160 |

## 4. Training: optimizer, learning rate, schedule

`src/stage1/train.py` (Stage 2 trainers follow the same pattern):

| what | where |
|---|---|
| learning-rate schedule: linear warm-up, then cosine decay to 0 | `lr_factor` <!--src/stage1/train.py::lr_factor-->L74–79 |
| EMA of the weights (diffusion, decay 0.999) | `EMA` <!--src/stage1/train.py::EMA-->L108–123 |
| one step: fp16 autocast, loss, backward, gradient clipping 1.0, optimizer + scheduler step | `train_step` <!--src/stage1/train.py::train_step-->L126–143 |
| validation on 256 fixed val cases; best.pt by tolerant mIoU | `validate` <!--src/stage1/train.py::validate-->L158–174 |
| checkpoints every 30 min + automatic resume (model, optimizer, scheduler, EMA, RNG) | `save_checkpoint` <!--src/stage1/train.py::save_checkpoint-->L202–207, `main` <!--src/stage1/train.py::main-->L221–375 |
| multi-GPU (DDP, 2 × T4 on Kaggle) | `src/distributed.py` |

| model | optimizer | lr | warm-up | steps × batch | config |
|---|---|---|---|---|---|
| Stage 1 matched U-Net | AdamW, wd 0.01 | 2e-4 | 1,000 | 30,000 × 64 | `configs/stage1/train_unet_matched.yaml` |
| Stage 1 diffusion (run 1) | AdamW, wd 0.01 | 1e-4 | 1,000 | 31,500 × 64 | `configs/stage1/train_diffusion.yaml` |
| Stage 1 constraint fine-tune / control | AdamW, wd 0.01 | 2e-5 | 200 | 26,000 × 64 | `configs/stage1/finetune_{constraints,control}.yaml` |
| Stage 2 parser v1 → v2B | AdamW, wd 0.01, encoder lr ⅓ | 3e-4 → 1e-4 | 500 → 200 | time budget 2.75 h → 1.85 h, batch 8 | `configs/stage2/parser_v1.yaml`, `parser_v2.yaml` |
| Stage 2 detectors v2 / v3 | AdamW, wd 0.01 | 1e-4 / 3e-4 | 100 / 200 | 2.0 h / 1.9 h | `configs/stage2/detector_v2.yaml`, `detector_v3.yaml` |
| Stage 2 16-class U-Net (K2a) | AdamW, wd 0.01 | 1e-4 | 200 | 3 h, batch 32 / GPU | `configs/stage2/train16_unet.yaml` |
| Stage 2 constrained / control diffusion (K1) | AdamW, wd 0.01 | 2e-5 | 200 | 24,000 × 64 | `configs/stage2/train16_{constrained,control}.yaml` |
| Stage 2 large-hole models (B4) | AdamW, wd 0.01 | 5e-5 / 2e-5 | 100 | 1.65 h, 256 px windows | `configs/stage2/train16_b4_*.yaml` |

All runs: gradient clipping 1.0, fp16 on the GPU, Kaggle T4 × 2.

## 5. Metrics, validation, selection, test

| what | where |
|---|---|
| prediction outside the mask replaced by the known plan before scoring (every method alike) | `src/stage1/eval/metrics.py` `composite` <!--src/stage1/eval/metrics.py::composite-->L57–59 |
| tolerant IoU (a wall / door / window pixel matches within 1 px) | `tolerant_counts` <!--src/stage1/eval/metrics.py::tolerant_counts-->L109–124 |
| component precision / recall / F1 (Hungarian matching, IoU ≥ 0.5) | `match_components` <!--src/stage1/eval/metrics.py::match_components-->L81–93 |
| per-case metrics incl. connectivity error, room leak, room count | `sample_stats` <!--src/stage1/eval/metrics.py::sample_stats-->L137–186 |
| architectural validity (no more rule violations than the ground truth) | `src/stage1/eval/arch_metrics.py` `case_metrics` <!--src/stage1/eval/arch_metrics.py::case_metrics-->L308–316 |
| model selection on val: Pareto front + knee | `src/stage1/selection.py` `pareto_front` <!--src/stage1/selection.py::pareto_front-->L99–103, `knee` <!--src/stage1/selection.py::knee-->L111–122 |
| paired bootstrap | `src/stage1/stats.py` `bootstrap` <!--src/stage1/stats.py::bootstrap-->L54–63 |
| the one test run (frozen settings) | `src/stage1/selection.py` `test_main` <!--src/stage1/selection.py::test_main-->L276–319 → `outputs/stage1/stage1_test.md` |

## 6. Stage 2 data

| what | where |
|---|---|
| 16 classes and the CubiCasa name mapping | `configs/stage2/classes16.yaml`, `src/stage2/data/classes16.py` |
| SVG → labelled polygons | `src/stage2/data/cubicasa_svg.py` `parse_svg` <!--src/stage2/data/cubicasa_svg.py::parse_svg-->L137–187 |
| **split**: the official CubiCasa5K lists; val → tuning / held-out | `src/stage2/data/build_cubicasa.py` `read_split_lists` <!--src/stage2/data/build_cubicasa.py::read_split_lists-->L53–61; `src/stage2/parse/val_split.py` |
| two rasterisations (parser set, completion set) | `src/stage2/data/build_cubicasa.py` `process` <!--src/stage2/data/build_cubicasa.py::process-->L70–167, `src/stage2/data/rasterise.py` |
| omissions simulated on the drawing (door kinds, window, wall, fixture, room; whole room, region) | `src/stage2/detect/image_eval.py` `omit` <!--src/stage2/detect/image_eval.py::omit-->L92–131; `src/stage2/data/erasure.py`, `erasure_large.py` |

## 7. PARSE

| what | where |
|---|---|
| network: segmentation-models-pytorch U-Net, ImageNet ResNet-34 encoder, greyscale in, 16 classes out | `src/stage2/parse/train.py` `Parser` <!--src/stage2/parse/train.py::Parser-->L36–47 |
| loss: class-weighted cross-entropy + soft Dice | `src/stage2/parse/losses.py` |
| black-and-white augmentations | `src/stage2/parse/augment.py` |
| v2: constraint fine-tune with gradient-size calibrated weights + text erasure | `src/stage2/parse/train_v2.py` `calibrate` <!--src/stage2/parse/train_v2.py::calibrate-->L122–145 |
| inference: scale choice (~8 px walls) and rescale to ~3 px | `src/stage2/pipeline/run.py` `parse` <!--src/stage2/pipeline/run.py::parse-->L129–149 |

## 8. CHECK

| what | where |
|---|---|
| constraint library C1–C18: one function per rule → per-pixel violation map, used as loss, detector input and verifier | `src/stage2/constraints/constraints.py` `violation_map` <!--src/stage2/constraints/constraints.py::violation_map-->L412–415, `constraint_loss` <!--src/stage2/constraints/constraints.py::constraint_loss-->L418–432 |
| which sources make up the final CHECK | `src/stage2/pipeline/run.py` `detect_parts` <!--src/stage2/pipeline/run.py::detect_parts-->L154–197; settings `configs/stage2/detector_threshold.yaml` |
| merging overlapping flags (type by evidence) | `src/stage2/detect/rules.py` `merge` <!--src/stage2/detect/rules.py::merge-->L216–238 |
| detector v3 (drawing + parse, ResNet-34 U-Net, 17 channels) | `src/stage2/detect/detector_v3.py`, training `src/stage2/detect/train_detector_v3.py` |
| how thresholds were chosen (rules first within 0.25 false alarms, detectors up to 0.4) | `src/stage2/detect/check_tune.py` |

## 9. FILL + VERIFY

| what | where |
|---|---|
| 16-class networks and loss (reconstruction + time-gated constraint terms) | `src/stage2/complete/models16.py` `Diffusion16.loss` <!--src/stage2/complete/models16.py::Diffusion16.loss-->L68–87 |
| warm start: 6 → 16 classes from the Stage 1 checkpoints | `src/stage2/complete/warm_start.py` `expand_state` <!--src/stage2/complete/warm_start.py::expand_state-->L38–64 |
| training loop | `src/stage2/complete/train16.py` |
| hybrid sampler: sketch → noise to t₀ → DDIM with known-region reset | `src/stage2/complete/sampler.py` `hybrid_sample` <!--src/stage2/complete/sampler.py::hybrid_sample-->L51–88 |
| VERIFY: violation counts + best-of-N | `violation_breakdown` <!--src/stage2/complete/sampler.py::violation_breakdown-->L91–113, `best_of_n` <!--src/stage2/complete/sampler.py::best_of_n-->L122–132 |
| large holes: B4 models on one big window | `src/stage2/complete/large_fill.py` `fill_large` <!--src/stage2/complete/large_fill.py::fill_large-->L244–258 |

## 10. End to end, final test, app

| what | where |
|---|---|
| the whole method | `src/stage2/pipeline/run.py` `run_pipeline` <!--src/stage2/pipeline/run.py::run_pipeline-->L490–550 |
| end-to-end scoring (found, fixed, false additions) | `src/stage2/pipeline/e2e_val.py` `score` <!--src/stage2/pipeline/e2e_val.py::score-->L136–171 |
| the one-shot test (FILL ablation, CHECK, end to end) | `src/stage2/final_test.py` `do_fill` <!--src/stage2/final_test.py::do_fill-->L98–128, `report` <!--src/stage2/final_test.py::report-->L271–348 |
| the app (UI only) | `app/app.py`; logic in `src/stage1/interface/` and `src/stage2/interface/demo.py` |

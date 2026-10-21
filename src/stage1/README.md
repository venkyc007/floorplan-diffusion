# src/stage1 — mask-given inpainting of ResPlan plans (128 × 128, 6 classes)

The final method is the **size-adaptive hybrid**: the matched U-Net draws the hole, the drawing is noised to
t₀ = 400 / 600 / 700 (small / medium / large hole) and the constraint fine-tuned diffusion model finishes it.
Deployed form: `interface/inference.py` (`Repairer.repair`); settings `configs/stage1/app.yaml`.

| part | files | what |
|---|---|---|
| data | `data/resplan.py`, `rasterize.py`, `classes.py`, `dedup.py`, `split.py`, `build_dataset.py`, `checks.py` | ResPlan → 128 × 128 maps, near-duplicate removal, leakage-free split |
| corruption + benchmarks | `data/corruptions.py`, `dataset.py`, `make_benchmark.py` | controlled removal by type / location / size; training dataset; fixed val / test benchmarks |
| models | `models/backbone.py`, `unet_baseline.py`, `diffusion.py`, `constraints.py` | shared UNet2DModel; U-Net inpainter (CE); Palette-style diffusion (x0, DDIM); clDice / closure losses |
| training | `train.py`, `../distributed.py` | AdamW, warm-up + cosine, fp16, EMA, checkpoints + resume, DDP |
| evaluation | `eval/metrics.py`, `eval/arch_metrics.py`, `eval/baselines.py`, `eval/figures.py`, `evaluate.py` | tolerant IoU, component F1, structure metrics, architectural validity |
| sampling + selection | `sampling_experiment.py` (`refine`, the hybrid sampler), `full_eval.py`, `arch_eval.py`, `postprocess.py`, `selection.py`, `stats.py` | every setting on val, Pareto / knee selection, paired bootstrap, the one test run |
| app logic | `interface/` | `inference.py`, `render.py`, `plan_io.py`, `report.py`, `suggest.py`, `random_mask.py`, `make_gallery.py` |
| diagnostics | `diagnose_diffusion.py`, `overfit_check.py`, `speed_test.py`, `visualize.py` | checks behind decisions in `docs/decisions.md` |

Summary of the stage: `docs/stage1_summary.md`; data details: `docs/data_notes.md`.

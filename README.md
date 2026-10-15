# Floor Plan Diffusion Reconstruction

Diffusion-based reconstruction of missing architectural floor plan components (walls, doors, windows,
rooms) on the ResPlan dataset. A plan is a 128 × 128 map with 6 classes (wall, door, window, room,
outdoor, background); the model gets a plan with a region masked out and redraws that region.

The recommended configuration (`outputs/selection.md`) is a **size-adaptive hybrid**: a U-Net drafts
the missing region, and a diffusion model fine-tuned with structural constraints re-noises and
finishes that draft. Larger holes are re-noised further (t₀ = 400 / 600 / 700 for small / medium /
large holes).

## Install

```bash
conda create -n floorplan python=3.11
conda activate floorplan
pip install -r requirements.txt
```

## Interface (Gradio)

```bash
python -m src.interface.make_gallery   # once: 24 val samples -> data/app/val_gallery.npz (needs the val benchmark)
python app/app.py                      # then open http://127.0.0.1:7860
```

- **Checkpoints**: the paths are in `configs/app.yaml` (the matched U-Net and the constraint fine-tune).
  Model files are not in git. Download them from the Kaggle kernel outputs (`docs/kaggle_workflow.md`).
  If one is missing, the app stops with a message naming it.
- **Device**: runs on CPU. A GPU is used if available. On a laptop CPU a diffusion repair takes about
  30–52 s (longer for larger holes). The U-Net takes about 1.4 s and nearest fill is instant
  (`outputs/interface/cpu_timing.md`; re-time on your machine with `python -m src.interface.time_methods`).
- **Memory**: checkpoints are loaded only when a method first needs them, so start-up is quick.
  With both models loaded, the app uses about 0.75 GB of RAM.
- **Without the dataset**: the app still works with the editor and uploads; only the sample gallery
  needs the val benchmark.
- Design and methodology: `docs/interface.md`. Screenshots: `outputs/interface/`.

## Reproducing the pipeline

| step | command | details |
|---|---|---|
| data | `python -m src.data.build_dataset` | `docs/data_notes.md` |
| benchmarks | `python -m src.data.make_benchmark` | fixed val/test corruptions |
| training (Kaggle GPUs) | `python -m src.train --config configs/<name>.yaml --out <run dir>` | `docs/kaggle_workflow.md` |
| evaluation | `python -m src.full_eval`, `python -m src.arch_eval`, `python -m src.stats`, `python -m src.selection` | results in `outputs/*.md` |
| tests | `python -m pytest tests/` | |

Every self-made choice is logged in `docs/decisions.md`.

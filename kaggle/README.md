# kaggle — how every model was trained and evaluated on Kaggle

Each folder holds `kernel-metadata.json` (Kaggle id, GPU or CPU, which datasets / earlier kernel outputs it
attaches) and `run.py` (what it runs). Launch: `kaggle kernels push -p kaggle/kernels/<stage>/<name>` (from the
project folder, after uploading the code dataset). Workflow, quotas and resume: `docs/kaggle_workflow.md`.
All kernels belong to the Kaggle account `venkatesh007c` and are private.

## Training (the models in `configs/stage2/pipeline.yaml` and `configs/stage1/app.yaml`)

| folder | Kaggle id | device | trains |
|---|---|---|---|
| `stage1/unet`, `stage1/unet_matched` | floorplan-unet, floorplan-unet-matched | GPU | Stage 1 U-Net baseline; matched U-Net (the final sketch model) |
| `stage1/diffusion` | floorplan-diffusion | GPU | Stage 1 diffusion run 1 |
| `stage1/finetune_constraints`, `stage1/finetune_control` | floorplan-finetune-* | GPU | Stage 1 constraint fine-tune (final) and its control |
| `stage2/stage2_data` | floorplan-stage2-data | CPU | Stage 2 datasets from CubiCasa5K (parser + completion sets, all splits) |
| `stage2/stage2_parser_v1`, `stage2/stage2_parser_v2` | floorplan-stage2-parser-v1 / -v2 | GPU | parser v1; parser v2A / v2B (v2B final) |
| `stage2/stage2_k1` | floorplan-stage2-k1 | GPU | 16-class diffusion: constrained (GPU 0) vs control (GPU 1), the ablation |
| `stage2/stage2_k2a` | floorplan-stage2-k2a | GPU | 16-class U-Net (+ detector v1) |
| `stage2/stage2_detv2_data`, `stage2/stage2_detv2` | floorplan-stage2-detv2(-data) | GPU | detector v2 (data from the parser, then training) |
| `stage2/stage2_detv3` | floorplan-stage2-detv3 | GPU | detector v3 (drawing + parse) |
| `stage2/stage2_b4` | floorplan-stage2-b4 | GPU | B4: U-Net + diffusion fine-tuned on 256 px windows with large holes |

## Evaluation

| folder | device | what |
|---|---|---|
| `stage1/full_eval`, `stage1/arch_eval`, `stage1/selection`, `stage1/sampling`, `stage1/diagnostic`, `stage1/timing` | GPU | Stage 1 predictions of every setting on val (scored on the laptop) |
| `stage1/stage1_test`, `stage1/stage1_test_score` | GPU / CPU | the one Stage 1 test run |
| `stage2/stage2_gates`, `stage2/stage2_eta` | GPU | constraint gates G2–G4; guidance strength |
| `stage2/stage2_eval_*`, `stage2/stage2_large_fill*`, `stage2/stage2_b4_val`, `stage2/stage2_judge` | CPU | Stage 2 val evaluations (CHECK tuning, held-out detection, end-to-end batches, large holes, B4 validation) |
| `notebooks/stage2_final_test` | GPU (T4 × 2) | **the one Stage 2 test run** (prepared, run once at the end: `docs/final_test_notebook.md`) |

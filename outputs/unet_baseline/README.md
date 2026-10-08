# U-Net baseline run: summaries

Copied from the Kaggle kernel output (venkatesh007c/floorplan-unet, 2026-10-08), without checkpoints (best.pt / last.pt stay in the git-ignored outputs/kaggle/unet/).

- train_log.csv: loss every 50 steps; val_log.csv: 256 fixed val cases during training
- eval_val.csv/.md: best.pt on the full val benchmark (7,772 cases)
- nearest_fill_val.csv: nearest-fill baseline on the same benchmark
- samples/: sample grids at each validation; config.yaml: run settings
- Summary: ../unet_baseline_results.md

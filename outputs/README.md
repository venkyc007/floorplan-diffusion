# outputs — results tables and figures

| folder | what |
|---|---|
| `stage1/` | Stage 1 (ResPlan, mask given). **`stage1_test.md` = the TEST result (run once)**; `selection.md` = the final choice on val; `full_eval.md`, `arch_eval.md`, `stats.md` = val evaluations; `baseline_results_val.md` = do-nothing / nearest-fill / ground-truth baselines on val; `baseline_results.md` = the Day-2 run of the same baselines on the test benchmark (before any model existed); run logs in `unet_baseline/`, `diffusion_run1/`, `finetune_runs/`, `unet_matched/`; app screenshots + CPU timing in `interface/` |
| `stage2/` | Stage 2 (CubiCasa5K drawings). Index with the data role of every file: `stage2/README.md` |
| `stage2/history/` | superseded Stage 2 results, kept because `docs/decisions.md` cites them |
| `kaggle/`, `runs/` | local copies of Kaggle outputs and checkpoints (git-ignored, about 7 GB) |

Every figure / table is produced by a script named in its first lines or in `docs/decisions.md`.

# Kaggle workflow

How training runs on Kaggle's free GPUs. All commands run from the project folder in the `floorplan` conda env. Kaggle account: `venkatesh007c`. Weekly GPU quota: 30 h (check with `kaggle quota`).

**Never print, copy or commit the API token** (`C:\Users\lenovo\.kaggle\access_token`). Don't run `kaggle config view`; it can print credentials.

## Pieces

| What | Kaggle ID | Local source |
|---|---|---|
| Data (private dataset) | `venkatesh007c/floorplan-data` | `python -m src.kaggle_prep data` → `data/kaggle_upload/floorplan-data/` (train + val label maps, metadata; no test data) |
| Code (private dataset) | `venkatesh007c/floorplan-code` | `python -m src.kaggle_prep code` → `data/kaggle_upload/floorplan-code/code.zip` (src/, configs/, requirements.txt) |
| Timing kernel | `venkatesh007c/floorplan-timing` | `kaggle/kernels/timing/` |
| U-Net training kernel | `venkatesh007c/floorplan-unet` | `kaggle/kernels/unet/` |
| Diffusion training kernel (training only, no full-val evaluation) | `venkatesh007c/floorplan-diffusion` | `kaggle/kernels/diffusion/` |
| Stage 2 data (CPU; CubiCasa5K → parser + completion sets) | `venkatesh007c/floorplan-stage2-data` | `kaggle/kernels/stage2_data/` (input: public `qmarva/cubicasa5k`). Later kernels attach it via `kernel_sources`. Download only the reports: `kaggle kernels output venkatesh007c/floorplan-stage2-data -p outputs/kaggle/stage2_data --file-pattern "report/"` |

Kernels are private Python scripts (`run.py` + `kernel-metadata.json`) with GPU and internet on. Each one:
1. copies the code and data from `/kaggle/input` into `/tmp/code`;
2. installs missing packages (e.g. diffusers);
3. rebuilds the val benchmark from its seed and checks it against `configs/benchmark_checksums.json`, stopping if it differs;
4. runs the project commands. Anything written to `/kaggle/working` becomes the kernel's output.

The setup logic lives in `src/kaggle_kernel.py`, inside the code bundle; each `run.py` only copies the code and imports it. Training goes through `kaggle_kernel.train()`, which uses `torchrun` with one process per GPU when Kaggle provides more than one (T4 ×2), and logs every GPU's utilisation to `gpu_usage.csv`.

**Don't push a new version of a kernel that is still running.** It can replace the running session; use a separate kernel ID for a parallel run.

**Windows CLI quirk:** `kaggle datasets create/version -p <relative/nested/path>` fails with a "No such file or directory … .kaggle/uploads" error. Run the command from **inside** the folder with `-p .` instead, as below.

## Upload a new code version

After changing anything in `src/` or `configs/`:

```bash
python -m src.kaggle_prep code
cd data/kaggle_upload/floorplan-code
kaggle datasets version -p . -m "short description of the change"
cd ../../..
kaggle datasets status venkatesh007c/floorplan-code      # wait for "ready"
```

New kernel runs use the latest version automatically. Data changes work the same way (`kaggle_prep data`, `floorplan-data` folder), but the data shouldn't change: the split is fixed.

## Launch a run

```bash
kaggle quota                                                   # enough GPU hours left?
cd kaggle/kernels/unet          # or kaggle/kernels/timing
kaggle kernels push -p . --accelerator NvidiaTeslaT4
cd ../../..
```

Each push starts a new **version** of the kernel and runs it from the top. Use `--accelerator NvidiaTeslaP100` if T4s are unavailable. A kernel can run for at most 12 h; training stops itself at `time_limit_hours` (8.5 h) and saves.

## Check status

```bash
kaggle kernels status venkatesh007c/floorplan-unet      # QUEUED / RUNNING / COMPLETE / ERROR
kaggle kernels logs venkatesh007c/floorplan-diffusion   # the log so far, also while the kernel runs
```

With two GPUs the log should show `process 0/2 on cuda:0 (Tesla T4)` and `process 1/2 on cuda:1 (Tesla T4)`, then `step N: loss ... (X samples/s)` every 50 steps. The 2-GPU timing run measured about 75 samples/s for diffusion and 78 for the U-Net.

Times: Kaggle and the CLI report UTC. For India time use the Windows clock (`Get-Date` in PowerShell); Git Bash's `TZ=Asia/Kolkata date` prints UTC on this machine.

On the website: <https://www.kaggle.com/code/venkatesh007c/floorplan-unet>. Open the running version to see the live log (training loss every 50 steps, validation scores). "Your Work" lists all kernels, and the Settings → Quotas page shows GPU hours used.

## Download outputs

```bash
kaggle kernels output venkatesh007c/floorplan-unet -p outputs/kaggle/unet
```

This downloads everything the kernel wrote to `/kaggle/working`, plus its log, to `outputs/kaggle/unet/`, which is git-ignored. For a training run that's `runs/unet/` with `checkpoints/{last,best}.pt`, `train_log.csv`, `val_log.csv`, `samples/`, `config.yaml` and `eval_val.csv/.md`.

## Resume from a checkpoint

Training resumes automatically when `runs/unet/checkpoints/last.pt` exists in its output folder. On Kaggle, each new session starts empty, so the previous output must be attached as an input:

1. On the finished kernel's page, use **Output → "New Dataset"** (or "Add to dataset"), creating a private dataset, e.g. `venkatesh007c/floorplan-unet-run1`.
2. Add it to `dataset_sources` in `kaggle/kernels/unet/kernel-metadata.json`.
3. Push the kernel again. `run.py` finds `runs/unet/checkpoints/last.pt` among the inputs, copies the run folder to `/kaggle/working`, and `src.train` continues from that step, with the best score, logs and cumulative hours carried over.

**Keep `max_steps` (and the batch size) unchanged when resuming.** The learning-rate schedule is computed from `max_steps`.

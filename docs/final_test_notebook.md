# Stage 2 final test: running it on a friend's Kaggle account

The final Stage 2 test runs once, as the Kaggle notebook `kaggle/notebooks/stage2_final_test/stage2_final_test.ipynb`.
It needs no API token and no internet: only three attached inputs. It uses both T4 GPUs and saves every finished case
as it goes (resume after a time-out). Everything it writes goes to `/kaggle/working`.

## What it evaluates (`src/stage2/final_test.py`, counts in `configs/stage2/test_eval.yaml`)

1. **FILL with the hole given, constrained vs control diffusion** (the ablation for the structural constraints):
   40 small + 40 medium + 40 large holes on `completion_test` windows. Same U-Net sketch and seed for both arms.
   Reported: tolerant mIoU, validity, violations per hole, fixed, with paired bootstrap 95% CIs of the difference.
2. **CHECK on test drawings**: 60 intact drawings (false alarms) + 120 with one size-balanced omission (a third each
   small / medium / large): found / right type by type, location, position, size bin.
3. **End to end on 60 more test drawings** (one size-balanced omission each): found, fixed, false additions,
   tolerant mIoU and validity in the omission region, time per drawing, by the same four fields.

Output: `final_test/final_test.md` (the report), `final_test/part0.jsonl`, `part1.jsonl` (one line per case),
`part*.log`, and `final_test.zip` (all of it).

## The inputs to share (all private, owner venkatesh007c)

| input | what | how the friend gets it |
|---|---|---|
| dataset `floorplan-code` | the project code (`python -m src.kaggle_prep code`, then upload a new version) | share with the friend (Viewer) |
| dataset `floorplan-stage2-final` | all checkpoints + offline pip wheels (`python -m src.stage2.package_final`, then `kaggle datasets create -p . -r zip` in `data/kaggle_upload/floorplan-stage2-final`) | share with the friend (Viewer) |
| notebook output `floorplan-stage2-data` | the CubiCasa test split: `parser_test.zip`, `completion_test.zip` | share that notebook with the friend (Viewer) |
| notebook `floorplan-stage2-final-test` (this one) | the evaluation | import the .ipynb on the friend's account (below), or share it as a collaborator (Editor) |

## Before sharing (me)

1. Upload the code dataset version that will be tested: `python -m src.kaggle_prep code`, then
   `cd data/kaggle_upload/floorplan-code` and `kaggle datasets version -p . -r zip -m "final test"`.
2. Build and create the checkpoint dataset: `python -m src.stage2.package_final` (it lists anything missing), then
   `cd data/kaggle_upload/floorplan-stage2-final` and `kaggle datasets create -p . -r zip`.
3. On kaggle.com share: the two datasets (Settings -> Sharing -> add the friend's username, Viewer) and the notebook
   `floorplan-stage2-data` (Share -> add the friend, Viewer).

## The friend's steps (on their own account)

1. kaggle.com -> **Create -> New Notebook -> File -> Import Notebook** -> upload `stage2_final_test.ipynb`.
2. Right panel: **Add Input** -> search and add `floorplan-code` and `floorplan-stage2-final` (Datasets, "Shared with
   you"), and the notebook output `floorplan-stage2-data` (Notebooks, "Shared with you").
3. Right panel -> **Session options**: Accelerator = **GPU T4 x2**; Internet can stay off; Persistence: none needed.
4. **Save Version** (top right) -> **Save & Run All (Commit)** -> Save. Close the tab if you like: it runs on Kaggle.
   About 3-5 GPU hours of the friend's quota (both GPUs count: ~2 x 2-2.5 h).
5. When the version shows "Complete": **Share** the notebook with `venkatesh007c` (Viewer is enough).
6. If it stops with a time-out (12 h limit, not expected): Add Input -> this notebook's own output (the last version),
   then Save & Run All again. It copies the finished cases back and continues.

## Getting the results (me)

- With the notebook shared to me: `kaggle kernels output <friend-username>/floorplan-stage2-final-test -p outputs/stage2/final_test`
  (or open the notebook -> Output -> download `final_test.zip`).
- Copy `final_test.md` into `outputs/stage2/` and commit it; it is the Stage 2 test result (run once, never again).

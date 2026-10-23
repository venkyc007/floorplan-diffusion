# Stage 2 final test: the one-shot run on the CubiCasa test split

**Status: DONE, run once on 2026-10-22** (code `7581cb2`, kernel `floorplan-stage2-final-test` v2, all 420 cases,
0 errors). Result: `outputs/stage2/final_test.md`; summary `docs/results.md` §0. Do not run it again.

The final Stage 2 test runs **once**, as the Kaggle notebook `kaggle/notebooks/stage2_final_test/stage2_final_test.ipynb`
(T4 × 2, no internet, no API token). It is the last step of the project: nothing may be tuned after it. Every case is
saved as it finishes, so a cut-off run resumes. Results go to `/kaggle/working/final_test/`.

## What it measures (`src/stage2/final_test.py`; counts in `configs/stage2/test_eval.yaml`)

1. **FILL with the hole given** on `completion_test` windows: 40 small + 40 medium + 40 large holes. Arms per case,
   same seed: the U-Net sketch alone; sketch + **constrained** diffusion; sketch + **control** diffusion (the
   constraint ablation, with paired bootstrap 95% CIs); for medium / large holes also the **shipped large-hole
   method** (B4 + big window). Tolerant mIoU, validity, violations per hole, fixed.
2. **CHECK** on test drawings: 60 intact (false alarms per drawing) + 120 with one size-balanced omission: found / right
   type, by type, location, position and size bin.
3. **End to end** on 120 more test drawings: found, fixed, false additions, tolerant mIoU and validity in the omission
   region, seconds per drawing, by the same four fields.

Sampling uses the **validated fast settings** (N = 2, 15 DDIM steps from t₀), the ones every val number was measured
with (`docs/decisions.md`, 2026-10-21 night). Expected time on T4 × 2: well under 1 hour.

## Inputs (all private, owner `venkatesh007c`)

| input | what | status |
|---|---|---|
| dataset `floorplan-code` | the project code at a committed version (`code_version.txt` inside) | needs a new version (upload) |
| dataset `floorplan-stage2-final` | the 10 checkpoints (1.6 GB) + offline wheels (smp 0.3.4, timm 0.9.7, diffusers 0.30.3, ...) | built locally in `data/kaggle_upload/floorplan-stage2-final/`; needs creating (upload) |
| notebook output `floorplan-stage2-data` | the CubiCasa splits: `parser_{val,test}.zip`, `completion_{val,test}.zip` | exists |

## Steps (each Kaggle step needs the user's go-ahead)

1. **Commit** the code (the version that will be tested) — done in the repo; check with `git status`.
2. **Upload the code version** (from the project folder):
   ```bash
   python -m src.kaggle_prep code
   ```
   then, inside `data/kaggle_upload/floorplan-code`:
   ```bash
   kaggle datasets version -p . -r zip -m "final test"
   ```
3. **Create the checkpoint dataset** (already built by `python -m src.stage2.package_final`), inside
   `data/kaggle_upload/floorplan-stage2-final`:
   ```bash
   kaggle datasets create -p . -r zip
   ```
   (about 1.6 GB; it also serves as the "trained models" deliverable — it can be shared with the TAs.)
4. **Smoke run on VAL (recommended, a few GPU minutes)**: set `SMOKE = True` in the notebook's first code cell, then
   push it: `kaggle kernels push -p kaggle/notebooks/stage2_final_test`. It runs 2 cases per section on the val split
   and never touches test. Check the log, then set `SMOKE = False`.
5. **The test run**: `kaggle kernels push -p kaggle/notebooks/stage2_final_test` with `SMOKE = False`
   (or on kaggle.com: open the notebook → Save Version → Save & Run All).
6. **Download**: `kaggle kernels output venkatesh007c/floorplan-stage2-final-test -p outputs/stage2/final_test`;
   copy `final_test.md` to `outputs/stage2/final_test.md`; commit. Never run it again.

GPU quota: `kaggle quota` (5.05 h were left on 2026-10-21; the run needs well under 1 h; the quota resets every
Saturday 05:30 IST). If it is cut off (12 h limit, not expected): attach the notebook's own last output as an input
and run again — finished cases are copied back and skipped.

**Running it on another Kaggle account** (if the quota is gone): share the two datasets and the
`floorplan-stage2-data` notebook with that account (Viewer), import the `.ipynb` there (File → Import Notebook), add
the three inputs, set Accelerator = GPU T4 × 2, Save & Run All, then share the notebook back and download its output.

**Without a GPU**: the same runner works on CPU (`STAGE2_DEVICE=cpu`, about 3–5 h on 4 cores), e.g. as one or two
Kaggle CPU sessions with `--part k --parts n`.

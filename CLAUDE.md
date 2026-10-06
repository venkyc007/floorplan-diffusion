# CLAUDE.md

Context for Claude Code. Read this at the start of every session.

## Project

**Diffusion-Based Reconstruction of Missing Architectural Floor Plan Components**
Solo course project. Started 2026-10-06, deadline ~2026-10-20 (about 2 weeks) — favor simple, working solutions over ambitious ones.

## Deliverables

1. **Dataset + controlled corruption** — floor plan dataset, plus code that removes components (walls, doors, windows, rooms) in a controlled way by *type*, *location*, and *size*.
2. **Diffusion model** — takes a floor plan with missing parts and reconstructs them.
3. **Structure-aware constraints** — wall connectivity, doors/windows lying on walls, axis alignment, room closure (as losses, guidance, and/or post-processing).
4. **Component-level evaluation** — per-class IoU, component detection precision/recall/F1, structural metrics (e.g. connectivity, closure, door/window-on-wall rate).
5. **Robustness + ablation study** — vary corruption type/size/amount; turn constraints and model parts on/off.

Also required: a **Gradio interface**, **trained models**, **documentation**, and a **stage-wise report**.

## Data

- **ResPlan** dataset (github.com/m-agour/ResPlan): ~17k unit-level vector floor plans. Raw files go in `data/raw/`, processed tensors in `data/processed/`.
- **Remove near-duplicates** (plans that are identical or almost identical) before splitting.
- **Make our own train/val/test split** so that no near-duplicate pair ends up in two different splits. Group similar plans together and assign each whole group to one split, so the same plan can't leak across splits.
- Pipeline: `python -m src.data.build_dataset [--limit 100] [--reuse-split]` (settings in `configs/data.yaml`). `--reuse-split` re-converts but keeps the existing duplicate removal + split — **use it for any re-conversion so the split never changes**. Outputs in `data/processed/`: `train/val/test.npz` (`masks` (N,6,128,128) uint8 one-hot, `plan_ids`), `metadata.csv` (all 17k plans incl. removed, with `split`, `duplicate_of`, `similarity_group`, scale columns), `similar_pairs.csv`. Sanity checks + figures: `python -m src.data.checks`.
- Result: 1,918 near-duplicates removed (sim ≥ 0.90; incl. 678 of 683 ResPlan "augmented" copies); plans with sim ≥ 0.85 kept in the same split → 12,066 / 1,508 / 1,508.
- ResPlan facts: coordinates are NOT metres — each plan is normalized to ~256 units on its long side (only `area` is metric). ~775 geometries are invalid, so rasterize by drawing coordinates (cv2), don't use shapely boolean ops. Load the pickle only via `src.data.resplan.load_plans` (allow-listed unpickler). `raw.githubusercontent.com` is blocked on the user's network — use `git clone` from github.com.
- Kaggle: upload `data/processed/` (~8 MB) as a private Kaggle dataset rather than re-running the pipeline there.
- `data/` is git-ignored; never commit dataset files.

## Representation

- Each plan is a **128×128 map with 6 channels**: `wall`, `door`, `window`, `room` (room interior), `outdoor`, `background`.
- Channel order is defined once in `src/data/classes.py` — always import it from there.
- Channels are mutually exclusive (one-hot: exactly one class per pixel). Paint order outdoor → room → wall → window → door, so doors/windows are drawn on top of walls.
- ResPlan mapping: door = door + front_door; room = enclosed indoor rooms only (living, bedroom, bathroom, kitchen, storage, stair); outdoor = balcony, garden, land (yards/setbacks/light wells), parking, pool (ResPlan has no terrace class); inner/neighbor → background.
- **Normalized per plan**: the building (walls, openings, rooms, balconies — not garden/land) is scaled so its longer side is 120 px (4 px padding), aspect ratio kept, centred; outdoor beyond the frame is clipped. Walls/doors are only ~2 px thick; doors/windows median ~27–30 px area, <0.1% under 6 px.
- **Real scale is preserved in metadata.csv**: `area_m2` (real area from the listing), `px_per_unit`, `metres_per_pixel` (= sqrt(area_m2 / (inner+wall area)) / px_per_unit), `wall_thickness_m`, and `scale_ok` (False for the 67 plans whose implied wall thickness is outside 0.10–0.40 m, i.e. a bad `area` value).

## Structural rules (for constraints and metrics)

- **Boundary = wall + door + window together.** Connectivity constraints/metrics treat all three as one boundary; doors and windows fill gaps in walls (they don't overlap them).
- **Gaps in interior walls without a door are legitimate open passages** (open kitchens, archways). Connectivity logic must tolerate them, not flag them as broken walls.
- **Room-closure constraints apply only to `room`, never to `outdoor`** (balconies/yards are open-sided by nature).

## Compute & budget

- Local machine: Windows laptop, 16 GB RAM, **no GPU**. Use local only for data prep, debugging on tiny subsets, evaluation, and the Gradio app.
- **Training runs on Kaggle free GPUs** (sessions time out, ~30 h/week quota). Therefore training code **must save checkpoints regularly and resume** from the latest one (model, optimizer, scheduler/EMA, epoch/step, RNG state).
- Budget is **zero** — free tools and services only.
- Be memory-conscious locally (16 GB): stream/process data in chunks, avoid loading the whole dataset at once.

## Environment

- Conda env: `floorplan` (Python 3.11). Install deps with `pip install -r requirements.txt`.
- On Kaggle, PyTorch is preinstalled — don't reinstall it there.
- Shell is Windows (PowerShell / Git Bash); use `pathlib` for paths so code works on both Windows and Kaggle (Linux).

## Project layout

```
src/        Python source code (the reusable logic: data, models, constraints, evaluation)
configs/    YAML config files (hyperparameters, paths, experiment settings)
notebooks/  Jupyter notebooks for exploration and Kaggle training runs
app/        Gradio web interface
data/       Datasets (git-ignored)
outputs/    Checkpoints, logs, figures, results (git-ignored except *.png figures)
```

## How to work with me (the user)

- **I'm new to coding.** For every file you create, explain in simple terms what it does and how it fits in.
- Keep code **modular** (small files, small functions, one job each) and **well commented**.
- Put settings in `configs/` rather than hard-coding numbers.
- Before big or hard-to-undo steps, explain the plan briefly first.

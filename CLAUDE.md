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
- **Exclude ResPlan's 683 "augmented" plans** (ids ≥ 100,000, listed in `split.json["augmented"]`). They are rotated/flipped/scaled copies of other plans, and ResPlan's augmentation **forgot to transform `inner`, `land`, `garden`, `parking`, `neighbor`** (verified: untransformed in 239/239 detectable cases) — so their outdoor areas and real scale are wrong.
- **Rooms are clipped to `inner`** (the indoor outline, +1 px): ~86 original plans label open-air areas outside the walls as rooms; those pixels become `outdoor`. Normal plans are unaffected (≤0.14% of room pixels).
- **Remove near-duplicates**, then **make our own train/val/test split** so that no near-duplicate pair ends up in two different splits. Dedup (see `src/data/dedup.py`): Stage 1 finds candidates by structure cosine (32×32, all 8 rotations/flips, ≥ 0.75); Stage 2 confirms with **tolerant mean per-class IoU** (wall/door/window/room/outdoor; a pixel matches if the other plan has the class within 1 px) on full 128×128 maps. A plan is removed only if tolerant IoU ≥ 0.90 with a **kept** plan (direct, no chain removals). Plain IoU was rejected: 2-px walls traced 1 px apart halve the wall/door IoU, so near-identical re-posts scored ~0.8. Kept plans with structure cosine ≥ 0.85 go in the same split (similarity groups).
- Pipeline: `python -m src.data.build_dataset [--limit 100] [--reuse-split]` (settings in `configs/data.yaml`). `--reuse-split` re-converts but keeps the existing exclusion + duplicate removal + split — **use it for any re-conversion so the split never changes**. Outputs in `data/processed/`: `train/val/test.npz` (`masks` (N,6,128,128) uint8 one-hot, `plan_ids`), `all_labels.npz` (label maps of all 17k plans, for checks), `metadata.csv` (all 17k plans incl. left-out ones; `split` ∈ train/val/test/removed_duplicate/excluded_augmented, `duplicate_of`, `duplicate_iou`, `similarity_group`, scale columns), `similar_pairs.csv` (candidates with cosine + tolerant IoU). Sanity checks + figures: `python -m src.data.checks`.
- ResPlan facts: coordinates are NOT metres — each plan is normalized to ~256 units on its long side (only `area` is metric). ~775 geometries are invalid, so rasterize by drawing coordinates (cv2), don't use shapely boolean ops. Load the pickle only via `src.data.resplan.load_plans` (allow-listed unpickler). `raw.githubusercontent.com` is blocked on the user's network — use `git clone` from github.com.
- Kaggle: upload `train/val/test.npz` + `metadata.csv` from `data/processed/` as a private Kaggle dataset rather than re-running the pipeline there (`all_labels.npz` is only needed for local checks).
- `data/` is git-ignored; never commit dataset files.

## Representation

- Each plan is a **128×128 map with 6 channels**: `wall`, `door`, `window`, `room` (room interior), `outdoor`, `background`.
- Channel order is defined once in `src/data/classes.py` — always import it from there.
- Channels are mutually exclusive (one-hot: exactly one class per pixel). Paint order outdoor → room → wall → window → door, so doors/windows are drawn on top of walls.
- ResPlan mapping: door = door + front_door; room = enclosed indoor rooms only (living, bedroom, bathroom, kitchen, storage, stair); outdoor = balcony, garden, land (yards/setbacks/light wells), parking, pool (ResPlan has no terrace class); inner/neighbor → background.
- **Framing rule — normalized per plan on the building's structure**: the bounding box of walls + doors (incl. front door) + windows is scaled so its longer side is 120 px (4 px margin each side), aspect ratio kept, centred. Nothing else affects the frame; balconies, outside stairs, garden/land/parking that extend beyond the 128×128 image are cropped. (Before 2026-10-07 balconies/rooms counted too, so a protruding balcony or outside stair shrank the building.) See `docs/data_notes.md`. Walls/doors are only ~2 px thick; doors/windows median ~27–30 px area, <0.1% under 6 px.
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
docs/       Documentation (docs/data_notes.md = data pipeline decisions + numbers)
data/       Datasets (git-ignored)
outputs/    Checkpoints, logs, figures, results (git-ignored except *.png figures)
```

## How to work with me (the user)

- **I'm new to coding.** For every file you create, explain in simple terms what it does and how it fits in.
- Keep code **modular** (small files, small functions, one job each) and **well commented**.
- Put settings in `configs/` rather than hard-coding numbers.
- Before big or hard-to-undo steps, explain the plan briefly first.

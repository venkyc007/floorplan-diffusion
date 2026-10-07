# Data notes

How the ResPlan dataset is turned into our training data, and why.
Code: `src/data/` · settings: `configs/data.yaml` · rebuild: `python -m src.data.build_dataset` · checks/figures: `python -m src.data.checks`.

## Source

- **ResPlan** (github.com/m-agour/ResPlan, data CC BY 4.0 — credit it in the report): 17,000 residential unit plans, one pickle (`ResPlan.pkl`), a list of dicts of shapely shapes.
- Coordinates are **not metres** (the README says otherwise): every plan is rescaled to ~256 units on its long side. Only `area` (m², from the listing) is metric.
- ~775 geometries are invalid (self-intersecting). We rasterize by drawing coordinates with OpenCV, which works on invalid shapes; avoid shapely boolean ops on raw data.
- The pickle is loaded with an allow-listed unpickler (`src/data/resplan.py`), because pickles can run code.

## Representation

128×128 image, 6 one-hot channels (exactly one class per pixel), defined in `src/data/classes.py`:

| Channel | ResPlan fields |
|---|---|
| wall | wall |
| door | door, front_door |
| window | window |
| room (enclosed indoor rooms) | living, bedroom, bathroom, kitchen, storage, stair — clipped to `inner` |
| outdoor | balcony, garden, land, parking, pool (no terrace class exists) |
| background | everything else (incl. `inner`, `neighbor`) |

Paint order: outdoor → room → wall → window → door, so doors/windows sit on top of walls (they fill wall gaps; they don't overlap walls in the raw data).

## Framing rule

Each plan is **normalized on its own building structure**:

1. Take the bounding box of **walls + doors (incl. front door) + windows** only.
2. Scale it so its longer side is **120 px** (4 px margin on each side), keeping the aspect ratio, and centre it.
3. Everything else — balconies, outside stairs, garden, land, parking — is drawn in that frame and **cropped** where it extends beyond the 128×128 image.

Why: earlier versions framed on walls + rooms + balconies, so a protruding balcony or an outside stair shrank the building (up to 1.8×). Switching to structure-only framing enlarged 4,769 plans by >1% (1,670 by >10%, 106 by >25%).

Consequence: tall or wide plans have empty margin on two sides; when the plan has a yard, that margin shows as outdoor. This is expected, not a framing error.

**Real scale is kept in `metadata.csv`**: `area_m2`, `px_per_unit`, `metres_per_pixel` (= sqrt(area_m2 / (inner + wall area)) / px_per_unit), `wall_thickness_m`, `scale_ok` (False for 67 plans whose implied wall thickness is outside 0.10–0.40 m, i.e. a bad `area` value). Median ≈ 0.11 m per pixel; walls/doors are ~2 px thick.

## Data problems found and how they are handled

1. **ResPlan "augmented" plans are broken → excluded (683 plans).** `split.json["augmented"]` lists rotated/flipped/scaled copies (ids ≥ 100,000). ResPlan's augmentation never transformed `inner`, `land`, `garden`, `parking` or `neighbor` (untransformed in 239/239 detectable cases), so their outdoor areas sit in the wrong place and their real scale is wrong. They are copies of plans already in the dataset anyway. Figure: `outputs/raw_vs_processed.png` (rows 1–2).
2. **Open-air areas labelled as rooms → clipped (~86 plans).** Some plans have room polygons outside the walls (e.g. plan 2093: a room covering the whole plot). Room pixels outside `inner` (+1 px) become outdoor. Normal plans change by ≤ 0.14% of room pixels.
3. **Known leftovers (not fixed, single plans):** 12117 has an oversized `land` shape; 4476 has an unusually large door polygon.

Frame-consistency check (original plans): median 100% of wall area lies inside `land` and of room area inside `inner`; augmented plans: median 77%.

## Near-duplicates and split

Listings are re-posted, so near-identical plans exist. Two stages (`src/data/dedup.py`):

1. **Candidates:** structure (wall+door+window) shrunk to 32×32, cosine similarity over all 8 rotations/flips, ≥ 0.75.
2. **Confirm:** **tolerant mean per-class IoU** on the full 128×128 maps (wall, door, window, room, outdoor; a pixel matches if the other plan has the same class within 1 px). Plain IoU was rejected: 2-px walls traced 1 px apart lose ~half their IoU, so near-identical re-posts scored ~0.8.

A plan is **removed only if its tolerant IoU ≥ 0.90 with a kept plan** (direct match, no chain removals; lowest plan id is kept). Kept plans with structure cosine ≥ 0.85 form **similarity groups**; a whole group goes to one split, so similar plans never cross train/val/test. The build asserts both rules.

## Current numbers (structure-only framing, 2026-10-07)

| | Plans |
|---|---|
| ResPlan total | 17,000 |
| Excluded augmented | 683 |
| Removed near-duplicates | 759 |
| **train / val / test** | **12,446 / 1,556 / 1,556** |

Doors: median 27 px per component, windows 30 px; < 0.1% are under 6 px.

Outdoor anomaly scan on the final 15,558 plans (pixel heuristics; they mostly flag legitimate yards, see `outputs/anomalies.png`): outdoor > 1.5× indoor: 3 · outdoor pixels > 10 px from any wall: 12,786 · outdoor box overlapping the indoor box < 10%: 248.

## Outputs

`data/processed/` (git-ignored): `train/val/test.npz` (`masks` (N,6,128,128) uint8, `plan_ids`), `all_labels.npz` (label maps of all 17,000 plans, for checks), `metadata.csv` (all plans; `split` ∈ train/val/test/removed_duplicate/excluded_augmented), `similar_pairs.csv`.

Figures in `outputs/`: `samples.png`, `opening_sizes.png`, `duplicate_pairs.png`, `anomalies.png`, `raw_vs_processed.png`.

# Decisions log

Choices made without asking (or decided by the user, marked), each with a one-line reason. Newest at the bottom of each section.

## Data (Day 1)

- **Augmented ResPlan plans excluded** — their `inner`/`land`/`garden`/`parking`/`neighbor` were never transformed (see `docs/data_notes.md`), and they are copies anyway.
- **Rooms clipped to `inner` (+1 px)** — some plans label open-air areas outside the walls as rooms; "room" must mean enclosed indoor space.
- **Parking and pool count as outdoor** — neither is an enclosed indoor room.
- **Tolerant IoU (1 px) for dedup** (user decision) — plain IoU halves the score of 2-px walls traced 1 px apart.
- **Frame = walls + doors + windows bounding box, 4 px margin** (user decision) — protruding balconies/stairs no longer shrink the building.

## Corruptions (Day 2)

- **Wall segments found by morphological opening** (horizontal and vertical runs of ≥ 6 wall px; pixels in both = corners/junctions, removed to split) — walls are axis-aligned and ~2 px thick, so this splits cleanly at corners, T-junctions and openings without a skeleton.
- **Doors/windows also end a wall segment** — they are not wall pixels, so a segment naturally stops at an opening.
- **Wall segment length = longer side of its bounding box, kept if 8–40 px** — matches the requested range; shorter pieces are stubs, longer ones are whole facades.
- **Room "with surrounding walls" = room component + wall/door/window pixels within 3 px** — walls are ~2 px thick, so 3 px catches the full surrounding boundary.
- **Rooms smaller than 20 px ignored** — slivers left by rasterization, not rooms.
- **Room components whose bounding box covers > 50% of the building are skipped** — rooms joined by open passages form one huge component (masks up to 155% of the building); removing that is "redraw the plan", not "one room".
- **Region = rectangle centred on a random building pixel, each side 8–64 px, clipped to the image; no extra padding** — a random rectangle carries no shape hint, so it needs no padding; 8–64 px spans door-sized to multi-room holes.
- **Mask padding: 2–6 px per side, independently, clipped to the image** — as specified; clipping keeps masks inside the 128×128 frame.
- **Location breakdown kept as is, although windows (99%) and rooms (97%) are almost always exterior** (user decision) — by nature: windows sit in outer walls and most rooms have an outer wall, so interior/exterior comparisons mostly reflect doors, walls and regions.
- **"Removed component" for location: the removed pixels (room: room + its boundary ring; region: building pixels in the rectangle)** — so a room on the facade counts as exterior because its outer wall is removed with it.
- **Exterior = removed pixels 4-adjacent to outdoor or background** — as specified; 4-adjacency matches the component connectivity.
- **Position: distance of the mask centroid from the building centroid ÷ building radius (largest centroid-to-building-pixel distance); edge if ≥ 0.55** — dividing by the radius makes it comparable across plan shapes; 0.55 is the median over 1,500 train plans × 5 types, so the two bins are balanced.
- **Building area = wall + door + window + room pixels** (outdoor excluded) — size should be relative to the apartment, not its yard.
- **Size bins: small < 2.7% ≤ medium < 7.7% ≤ large (mask area % of building area)** — the tertiles (2.69%, 7.71%) of single-corruption masks pooled over 1,500 train plans × 5 types.
- **Corrupted label map marks masked pixels as 255 ("unknown")** — one compact uint8 array carries both the known pixels and the mask.
- **`n_components` = number of parts removed (1, or 2–3 for multi)**; region counts as 1 — a rectangle is one removal operation.

## Training dataset (Day 2)

- **Multi-removal: types drawn independently with the same type probabilities (repeats allowed), masks united; overlapping masks allowed** — simplest unbiased reading of "mixed types".
- **If a requested type has nothing to remove, fall back to the other types in random order and count the skip** — training can't skip a sample; the counts stay visible in `dataset.skips`.
- **Rotation/flip applied to the label map before corrupting** — as specified; corruptions are then drawn on the rotated plan.
- **Labels kept in memory as uint8 label maps (204 MB), one-hot built per sample** — 6× less memory than one-hot arrays.
- **Per-process random generator seeded from `torch.initial_seed()`** — DataLoader workers differ, and `torch.manual_seed` makes runs repeatable.
- **One OpenCV thread per worker process** — otherwise OpenCV's threads oversubscribe the CPU when several workers run.
- **Returned tensors are float32 as specified** — transfer between worker processes is the speed limit on Windows (~1,200 samples/s with 4 workers); if it ever limits GPU training, returning uint8 and converting on the GPU would cut transfer 4×.

## Benchmarks (Day 2)

- **One sample per (plan, type); the random seed is derived from (2026, plan id, type index)** — each sample is reproducible on its own, independent of processing order.
- **Saved as label maps (`inputs` = corrupted map with 255 in the mask, `targets`, `masks`), not 7-channel float arrays** — 7,774 × 7 × 128 × 128 floats would be ~3.5 GB; `corruptions.to_model_input()` rebuilds the 7-channel input exactly.
- **Stored in `data/benchmarks/` (git-ignored)** — fully reproducible from the seed with `python -m src.data.make_benchmark`.

## Metrics (Day 2)

- **Prediction outside the mask is replaced by the known pixels before scoring** — the model is given them; it should be neither rewarded nor punished there.
- **Per-class IoU pooled over the benchmark (Σ intersection / Σ union); mIoU = per-sample mean over classes present in pred or GT, then averaged over samples** — pooled IoU is the standard segmentation number; per-sample mIoU follows the spec's "exclude absent classes" rule.
- **Tolerant IoU for wall, door, window (user decision), reported next to the strict IoU** — a pixel matches if the other map has the same class within 1 px (as in dedup); tolerant IoU = matches / (total − matches), pooled over the benchmark. Plain IoU halves for 2-px walls drawn 1 px off.
- **Tolerant IoU compares only pixels inside the mask, on both sides** — otherwise known walls just outside the mask "match" true walls at the mask edge, and a model that draws nothing scored 0.07 (a unit test caught this).
- **Tolerant mIoU = per-sample mean using tolerant IoU for wall/door/window and strict IoU for room/outdoor** — tolerance matters for thin classes; for large areas it would only blur the score.
- **Pixel accuracy counts all 6 classes inside the mask (incl. background)** — background inside the mask is a real answer.
- **Component matching uses each component's full extent (not only the part inside the mask)** — composited pixels outside the mask are identical in pred and GT, so this measures the whole door/room.
- **Boundary components counted with 8-connectivity; door/window/room components with 4-connectivity** — wall pixels touching only diagonally are still one structural line; openings and rooms follow the spec's 4-connectivity.
- **Room leak considers rooms overlapping or bordering the mask (mask grown 1 px)** — a missing wall in the mask opens the neighbouring rooms even if no room pixel is drawn inside it (a unit test caught this).
- **Room count error = |#enclosed regions pred − gt| (user decision; replaces room-pixel components)** — regions = 4-connected components of non-boundary pixels that do not touch the image edge (edge-touching ones are the outside world: yards, open balconies, background). A hole in a wall now merges the rooms on both sides, so leaving holes is penalised (do-nothing 1.99 vs nearest fill 1.70; with room-pixel components do-nothing "won", 0.29 vs 0.91). Errors above 1 are real: a padded door mask often also cuts nearby walls and opens 3+ rooms at once. Courtyards/light wells fully enclosed by walls count as regions in both pred and GT. Tiny wall-gap pockets are negligible (6 of 500 test plans have one < 10 px).
- **Openings-on-wall and room leak are pooled ratios (Σ hits / Σ components); conn_err and room_count_err are per-sample means** — rates should weight every component equally; errors are per plan.
- **Alignment and openings-on-wall kept as is** (user decision) — they barely separate nearest fill from ground truth (it copies straight lines), but are meant to catch blobby/diagonal model output.
- **Alignment score = share of predicted boundary pixels inside the mask that lie on a horizontal or vertical run of ≥ 5 boundary pixels (morphological opening with 1×5 and 5×1 lines)** — straight axis-aligned walls are fully covered, diagonal/staircase or blobby strokes are not; ground truth scores 1.00. Undefined (NaN) when the prediction draws no boundary in the mask.
- **Rates with nothing to count (e.g. no openings predicted) are NaN, not 0 or 1** — "no data" must not look like a perfect or a failing score.
- **Markdown tables written by a small helper instead of pandas' `to_markdown`** — avoids an extra dependency (tabulate).
- **pytest added to requirements** — needed to run `tests/`.
- **`outputs/*.md` committed like `outputs/*.png`** — `baseline_results.md` is a report artifact; the larger CSV stays ignored.

## Tooling

- **Always pass `encoding="utf-8"` when Python reads/writes text files** — Windows defaults to cp1252; a write with "−" failed and emptied this file once (rebuilt from the session record).

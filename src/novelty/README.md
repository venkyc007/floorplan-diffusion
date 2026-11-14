# `src/novelty/` - large omissions, end to end (novelty assessment)

The shipped Stage 2 pipeline fixes small omissions (a door, a window) but almost no large ones (a whole room or
region erased from the drawing). This folder adds a path for large omissions and joins it to the shipped pipeline
as one hybrid. Results and figures: `docs/novelty.md`; every choice and its reason: `docs/decisions.md`
("Novelty assessment"). Nothing in `src/stage1/` or `src/stage2/` changes behaviour: `run_pipeline` only gained two
optional arguments that are off by default.

| file | what it does | how it fits in |
|---|---|---|
| `build_set.py` | Erases a large part of each training drawing (the evaluation's own generators), runs the pipeline's parser on it and stores parsed map + ground truth + omission map. | The training data of everything below: it is exactly what the pipeline sees. Kaggle kernel `kaggle/kernels/novelty/large_data`. |
| `data.py` | Cuts training samples from that set: 256 px windows for the fill models (parser's map as context, true layout in the hole), whole-plan canvases for the locator and the structure net. Also the ink map and the stroke map of a drawing (round 2). | Used by `train_fill.py`, `locate.py`, `structure.py`. |
| `rays.py` | Round 2. Wall-continuation rays: from every end of a straight wall, the line it would follow if it simply went on to the next wall. | Two input channels of the structure net. |
| `structure.py` | Round 2. The structure net: parsed map + rays + ink map of the whole plan in, "a wall is missing here" out; decoding to 3 px walls, grouping them into region flags; training. | Kernels `structure`, `structure_abl` (without ink / rays), `structure2` (with the stroke map). Gives the final pipeline its region flags and the walls painted into the fill. |
| `ffc.py` | Fourier context blocks (after LaMa): a residual block that mixes information across the whole image in one step. Starts as the identity. | Optional part of the fill models (`models.py`); ablation with / without. |
| `models.py` | Builds the large-hole U-Net / diffusion (the Stage 2 16-class models + the Fourier blocks), warm start from B4, loading. | Used by the trainer and the pipeline. |
| `train_fill.py` | Trains a large-hole fill model (U-Net or diffusion). Same recipe as B4, new data. | Kernels `fill_ffc`, `fill_plain` (ablation). |
| `locate.py` | The region locator: the parser's map of the whole plan in, "a large part is missing here" probability out; training and inference. | Kernel `locate`; gave the round-1 pipeline its region flags (off in the final config). |
| `pipeline.py` | The hybrid: shipped pipeline + structure net (region flags, walls) + large-hole fill. `run(image)` / `run(image, boxes=...)`. | Entry point; settings in `configs/novelty/pipeline_structure_c.yaml` (v3c, the final hybrid; the default). Earlier versions run with the environment variable `NOVELTY_CONFIG`: `pipeline_structure_b.yaml` (v3b, structure net without the stroke map), `pipeline_structure.yaml` (v3a), `pipeline_large_edit.yaml` (v2, round-1 final), `pipeline_large.yaml` (v1), `pipeline_large_plain.yaml` (no Fourier blocks). |
| `eval_maps.py` | Tuning drawings, map level: locator threshold sweep, fills with the hole given, locator -> fill. | Where settings are chosen. |
| `eval_e2e.py` | Drawings, end to end: shipped vs hybrid on small / medium / large omissions and intact drawings. | The reported numbers (held-out val drawings). |

Order of a full run: `build_set` (Kaggle) -> `locate` + `train_fill` + `structure` (Kaggle) -> `eval_maps` and
`eval_e2e --split tuning` (choose settings on tuning drawings) -> freeze the pipeline config ->
`eval_e2e --split heldout` (report).

Two fill versions exist. v1 hides the flagged area from the fill models (inpainting, as Stage 1 / B4). v2, the
final one, keeps the parser's map visible there and lets the models correct it (edit mode, `configs/novelty/fill_*_edit.yaml`);
it was chosen over v1 on the tuning drawings. Round-1 result (151 held-out drawings): large omissions fixed
13% -> 36%, recovered within 2 px 11% -> 52%, small omissions unchanged (`docs/novelty.md` section 5).

Round 2 ("structure first", the final hybrid) keeps the v2 fill and replaces the locator by the structure net:
it marks the missing walls on the whole plan, groups of them are the region flags, and they are painted into the
fill's sketch; the fill also draws a wall wherever its wall probability is at least 0.18. Result on the same
held-out drawings, first version (v3b): large omissions recovered within 2 px 75%, fixed 48%. The final version
(v3c) gives the structure net one more input, the stroke map, so that it also sees the walls of faint drawings:
recovered 80%, fixed 55%, small omissions 41% (shipped 42%), a region flag on 3% of intact drawings
(`docs/novelty.md` section 8). The gain depends on the ink map (without it 55% recovered on the tuning
drawings): it shows the blank area a large omission leaves, not traces of the erased walls.

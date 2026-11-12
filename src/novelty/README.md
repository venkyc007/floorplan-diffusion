# `src/novelty/` - large omissions, end to end (novelty assessment)

The shipped Stage 2 pipeline fixes small omissions (a door, a window) but almost no large ones (a whole room or
region erased from the drawing). This folder adds a path for large omissions and joins it to the shipped pipeline
as one hybrid. Results and figures: `docs/novelty.md`; every choice and its reason: `docs/decisions.md`
("Novelty assessment"). Nothing in `src/stage1/` or `src/stage2/` changes behaviour: `run_pipeline` only gained two
optional arguments that are off by default.

| file | what it does | how it fits in |
|---|---|---|
| `build_set.py` | Erases a large part of each training drawing (the evaluation's own generators), runs the pipeline's parser on it and stores parsed map + ground truth + omission map. | The training data of everything below: it is exactly what the pipeline sees. Kaggle kernel `kaggle/kernels/novelty/large_data`. |
| `data.py` | Cuts training samples from that set: 256 px windows for the fill models (parser's map as context, true layout in the hole), whole-plan canvases for the locator. | Used by `train_fill.py` and `locate.py`. |
| `ffc.py` | Fourier context blocks (after LaMa): a residual block that mixes information across the whole image in one step. Starts as the identity. | Optional part of the fill models (`models.py`); ablation with / without. |
| `models.py` | Builds the large-hole U-Net / diffusion (the Stage 2 16-class models + the Fourier blocks), warm start from B4, loading. | Used by the trainer and the pipeline. |
| `train_fill.py` | Trains a large-hole fill model (U-Net or diffusion). Same recipe as B4, new data. | Kernels `fill_ffc`, `fill_plain` (ablation). |
| `locate.py` | The region locator: the parser's map of the whole plan in, "a large part is missing here" probability out; training and inference. | Kernel `locate`; gives the pipeline its region flags. |
| `pipeline.py` | The hybrid: shipped pipeline + locator + new large-hole fill. `run(image)` / `run(image, boxes=...)`. | Entry point; settings in `configs/novelty/pipeline_large.yaml`. |
| `eval_maps.py` | Tuning drawings, map level: locator threshold sweep, fills with the hole given, locator -> fill. | Where settings are chosen. |
| `eval_e2e.py` | Drawings, end to end: shipped vs hybrid on small / medium / large omissions and intact drawings. | The reported numbers (held-out val drawings). |

Order of a full run: `build_set` (Kaggle) -> `locate` + `train_fill` (Kaggle) -> `eval_maps` (choose settings on
tuning drawings) -> freeze `configs/novelty/pipeline_large.yaml` -> `eval_e2e --split heldout` (report).

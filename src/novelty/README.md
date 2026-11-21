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
| `pipeline.py` | The hybrid: shipped pipeline + structure net (region flags, walls) + large-hole fill. `run(image)` / `run(image, boxes=...)`. | Entry point; settings in `configs/novelty/pipeline_sound_select_c.yaml` (v4c, the round-4 final: structurally sound fills; the default). Other versions run with the environment variable `NOVELTY_CONFIG`: `pipeline_structure_c.yaml` (v3c, round-2 final, accuracy first), `pipeline_valid.yaml` (v3d, + clean-up), `pipeline_sound.yaml` (v4, + repair), `pipeline_sound_select.yaml` (v4b, + selection by soundness), `pipeline_sound_select_ck.yaml` (v4c-keep), ablations `pipeline_abl_sketch.yaml` / `pipeline_abl_n1.yaml`, `pipeline_structure_b.yaml` (v3b, structure net without the stroke map), `pipeline_structure.yaml` (v3a), `pipeline_large_edit.yaml` (v2, round-1 final), `pipeline_large.yaml` (v1), `pipeline_large_plain.yaml` (no Fourier blocks). |
| `eval_maps.py` | Tuning drawings, map level: locator threshold sweep, fills with the hole given, locator -> fill. | Where settings are chosen. |
| `walkthrough.py` | Step-by-step pictures of the final hybrid on the first held-out large cases (no picking): original drawing, drawing given, ink map, parser's map, structure net's walls + region flag, fill, the result drawn back on the drawing, ground truth, scores (`--tag _valid` with `NOVELTY_CONFIG=configs/novelty/pipeline_valid.yaml` for v3d). | A viewer for the report / slides; changes no setting. Pictures in `outputs/novelty/walkthrough/`. |
| `regularise.py` | Round 3 (validity). The clean-up of a region fill: added walls become straight pieces of one thickness that meet the existing walls; curved pieces, blobs, specks and loose pieces are dropped; a closed new room gets one room type. Also the validity counts (`checks`). | Called at the end of `pipeline.fill_region` when the config has `regularise.use: true` (`configs/novelty/pipeline_valid.yaml`, v3d). No model, no training. |
| `validity_report.py` | Reads the evaluation's CSV rows and prints accuracy next to validity for every version, with the true layout as the reference line. | The table of `docs/novelty.md` section 9. |
| `soundness.py` | Round 4. The three structural checks of the review on a completed region, without the true layout: walls connect (no dead ends, no floating groups), openings sit in walls, rooms are closed (not open to the outside, reachable through a door); "sound" = all three. Also a map of where each problem is. | Columns `s_*` (plan), `sp_*` (parser's map), `so_*` (oracle), `st_*` (true layout) of `eval_e2e.py`; acceptance test of `repair.py`; the violation maps of `assessment_figures.py`. |
| `repair.py` | Round 4. Repairs a region fill after the clean-up so that it passes those checks: joins floating walls, closes rooms open to the outside, finishes or removes openings, extends dead ends, gives every closed room a door, repaints loose room-type patches in open areas. A change is kept only if the problems go down. | Called at the end of `pipeline.fill_region` when the config has `repair.use: true` (`configs/novelty/pipeline_sound.yaml`, v4). With `select.by: soundness` (`pipeline_sound_select.yaml`, v4b) the fill finishes 4 samples and keeps the soundest. No model. |
| `soundness_report.py` | Tables: accuracy next to the three checks and the rule check (in four groups, with "valid vs oracle") for every version, with the parser's map, the oracle and the true layout as reference lines; paired differences. | `outputs/novelty/soundness_*.md`. |
| `completion_by_size.py` | The completion rate broken down by omission size, with the definition of a completed omission (review request); re-reads saved per-case results only. | `outputs/novelty/completion_by_size.md`. |
| `assessment_figures.py` | Pictures for the review: (1) corrupted plan, reconstruction and violation maps together for the first K large cases; (2) one plan with each use of the constraints switched on in turn; (3) several reconstructions of one omission, each with its checks (plausible alternatives). | `outputs/novelty/assessment/`. A viewer only. |
| `eval_e2e.py` | Drawings, end to end: shipped vs hybrid on small / medium / large omissions and intact drawings; per case also the validity counts, the three structural checks (plan, parser's map, oracle, truth) and the rule check rule by rule. | The reported numbers (held-out val drawings). |
| `app_demo.py` | The Gradio app's novelty tab: runs one frozen version (v4c / v4b / v3c, `configs/novelty/app.yaml`) on a drawing, draws the structural problems left in the filled regions, scores gallery cases with `eval_e2e.row_of`, side-by-side comparison. | `demo_app/app.py`. A viewer only. |
| `make_gallery.py` | Builds the novelty tab's example gallery: the held-out evaluation's own cases (same order, generator and seeds), with what each version did on them. | `data/app/novelty_gallery/` (git-ignored). |

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

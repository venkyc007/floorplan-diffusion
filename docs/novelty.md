# Novelty assessment: large omissions, end to end

Material for the novelty assessment (numbers, tables, figures, what each part does). It is not report prose.
Every number names its file in `outputs/novelty/`; every choice and its reason is in `docs/decisions.md`
("Novelty assessment"). Code: `src/novelty/` (its README lists the files). **Status: two rounds done. The final
hybrid is the round-2 one ("structure first", v3b, section 8), chosen on tuning drawings and evaluated once on the
held-out drawings; test split not used (section 6).** Sections 1-5 describe round 1 (locator + retrained fill, v1 / v2)
as it was run; section 8 describes round 2, which builds on it.

**Both rounds in one table** (151 held-out val drawings, nothing about the omission given, large omissions;
`outputs/novelty/e2e_heldout_round2.md`):

| pipeline | flagged as a region | recovered (2 px) | fixed (strict) | intact drawings with a region flag |
|---|---|---|---|---|
| shipped Stage 2 | 3% | 11% [7, 17] | 13% [8, 19] | 1% |
| round 1 final (v2: locator + edit fill) | 80% | 52% [44, 59] | 36% [29, 44] | 13% |
| **round 2 final (v3b: structure net + edit fill)** | **91%** | **75%** [67, 81] | **48%** [40, 56] | **7%** |

"Recovered" and "fixed" are defined in section 3; a perfect answer (the true layout pasted in) scores 96% recovered
but only 72% fixed on the tuning cases, so 48% fixed is two thirds of what the strict rule can give.

## 1. The problem, in the shipped system's own numbers

A "large omission" is a whole room, or a rectangle covering 5-25% of the building, missing from the drawing
(`src/stage2/data/erasure_large.py`): the interior walls, doors, windows and fixtures of that area are gone, and
the drawing looks like a plan that never had them. Nothing is given to the system: it must notice that a part is
missing, decide where, and draw it.

Shipped pipeline (frozen Stage 2) on the 151 held-out val drawings, one omission per case
(`outputs/novelty/e2e_heldout_base.md`):

| omission size | found (any flag) | flagged as a region | fixed, automatic | fixed, box given by hand |
|---|---|---|---|---|
| small (151) | 61% | 1% | 42% | - |
| medium (151) | 73% | 1% | 15% | 47% |
| large (151) | 80% | 3% | 13% | 35% |

So the shipped CHECK notices most large omissions, but as small symptoms (a door, a sink): it almost never says
"a region is missing", so the large-hole fill never runs (13-15% fixed). Even with the box given by hand, the
shipped large-hole models (B4) fix 35-47%.

Two causes, both measured before any new training:

1. **No region flag.** The one region rule (R8) fires on ~1-3% of large omissions.
2. **The fill models never saw a parser's map.** B4 was trained on clean ground-truth label maps (29 / 30 large
   holes fixed there) but is used on the parser's map of the erased drawing (6 / 20 on real drawings,
   `outputs/stage2/large_fill_compare_v2.md`).

## 2. What was built

```
drawing -> PARSE -> CHECK (shipped rules + detectors) -> small flags  -> shipped small-hole fill   (unchanged)
                 -> LOCATE (new: whole-plan locator)  -> region flags -> new large-hole fill       (new)
```

| part | what it is | file |
|---|---|---|
| Large-omission set | Every training drawing with a large part erased (the evaluation's own generators), run through the pipeline's own parser: parsed map + ground truth + omission map. 3,944 train drawings / 15,721 versions; 212 tuning drawings / 635 versions. | `src/novelty/build_set.py` |
| LOCATE | A U-Net (ImageNet ResNet-34 encoder) that reads the parser's 16-class map of the WHOLE plan and outputs the probability "this area is missing". It never sees the drawing, so it cannot learn erasure smudges. Two seeds averaged. | `src/novelty/locate.py` |
| Large-hole FILL | The Stage 2 16-class U-Net + diffusion hybrid (sketch -> re-noise -> diffusion finish -> best of N), retrained on the large-omission set: context = the parser's map, target = the true layout. | `src/novelty/train_fill.py`, `models.py` |
| Edit mode (fill v2, the final one) | The fill models SEE the parser's map inside the flagged box and correct it, instead of having it blanked and redrawing it blind (inpainting, v1). The merged room still shows hints of the missing one. Half of the training holes are the locator's own boxes. | `configs/novelty/fill_*_edit.yaml`, `data.to_model(hide=False)` |
| Fourier context blocks | Residual FFT blocks (after LaMa) at the three deepest stages of both networks: every output pixel sees the whole window in one step. Zero-initialised, so the warm-started network starts exactly as B4. | `src/novelty/ffc.py` |
| Hybrid pipeline | The shipped pipeline with two switch-in points: region flags from LOCATE (they swallow the small flags inside them) and the new fill for region flags / large manual boxes. Small omissions take the shipped path untouched. | `src/novelty/pipeline.py`, `configs/novelty/pipeline_large_edit.yaml` (final; `pipeline_large.yaml` = v1) |

The frozen Stage 2 settings are not edited; `run_pipeline` got two optional arguments that default to off.

## 3. How it is judged

- **Choose on tuning, report on held-out.** Every setting (locator threshold, re-noising step, which fill version)
  is chosen on the 212 TUNING val drawings. The 151 HELD-OUT val drawings are used for the reported comparison only.
  The Stage 2 test split is not touched (a guarded `--split test` run is prepared for one approved run).
- **Same cases, same scorer as the Stage 2 test.** Per held-out drawing: the intact drawing, and one small, one
  medium and one large omission (`omission_mix.omit_in_bin`); scored by `e2e_val.score`, the function behind the
  Stage 2 test table. Shipped and hybrid pipelines get exactly the same erased drawings.
- **Fairer columns for large fills** (the report's section 8.1 promised them):
  - *recovered (2 px)*: the omission is found, at least half of the truly missing wall pixels are back within 2 px,
    and at least half of the added wall pixels lie within 2 px of a true wall. "Fixed" uses strict pixel IoU, which
    is ~0 for a 3 px wall drawn 2 px off, and it also counts structure that was never missing.
  - *ceiling*: the true layout pasted into the parser's map. On the tuning cases even this scores only 72% "fixed"
    and 2% "valid" (the seam between truth and parser's map breaks rules; erased fixtures and doors are not walls),
    while it is 96% "recovered". So validity is reported as violations per hole, not as a pass rate.
  - *oracle best of N*: the best of the N samples by wall IoU.

## 4. Results on the tuning drawings (used to choose; optimistic by construction)

### 4.1 LOCATE (`outputs/novelty/tune_locator_v1.md`; 423 erased versions + 212 intact drawings, map level)

| setting | large omissions found | whole rooms | erased rectangles | region flags per intact drawing |
|---|---|---|---|---|
| shipped region rule R8 (held-out, for reference) | 1-3% | | | 0.01 |
| locator, one model, threshold 0.5 | 69-73% | | | 0.14-0.16 |
| **locator, 2-model average, threshold 0.4 (chosen)** | **75%** | **83%** | **67%** | **0.18** |
| locator, 2-model average, threshold 0.5 | 67% | | | 0.11 |

Rule for the threshold (set before the sweep was read): the highest recall with at most 0.2 region flags per intact
tuning drawing (half of the shipped CHECK's 0.4 budget). Whole rooms are found more often the larger they are
(67% below 5% of the building, 93% above 12%). The outline of the flag matches the missing room well (pixel IoU
0.68 when found).

A second filter was tried and rejected: keep a region flag only if the fill model would draw at least G new wall
pixels there (`outputs/novelty/tune_gate_v1.md`). It removes true and false flags at about the same rate
(e.g. 75% -> 65% found for 0.18 -> 0.14 false flags), so it stays off.

### 4.2 FILL with the hole given (`outputs/novelty/tune_maps_v1.md`; 120 erased versions, map level)

| fill | recovered (2 px) | missing-wall recall | added-wall precision | fixed (strict) | tol. mIoU in the box | violations per hole |
|---|---|---|---|---|---|---|
| nothing (the parser's map) | 0% | 0.00 | - | 6% | 28.3 | - |
| shipped B4 | 37% | 0.46 | 0.56 | 40% | 29.9 | 17.6 |
| new U-Net sketch alone | 47% | 0.57 | 0.62 | 38% | 33.9 | 33.1 |
| **new hybrid, t0 700** | **46%** | 0.53 | 0.62 | 38% | 33.5 | 17.7 |
| new hybrid, t0 900 | 48% | 0.53 | 0.58 | 38% | 33.0 | 16.9 |
| ceiling (truth pasted) | 96% | 1.00 | 1.00 | 72% | 100 | 19.8 |

Reading: retraining on the parser's maps brings back clearly more of the missing walls (recall 0.46 -> 0.53-0.57,
recovered 37% -> 46-48%) and the diffusion finish halves the sketch's rule violations (33.1 -> 17.7) at the same
accuracy, as in Stage 1 and Stage 2. Strict "fixed" does not move (38-40%): it is dominated by structure that was
never missing. The re-noising step barely matters (t0 700 kept: the Stage 1 / B4 value for large holes).

### 4.3 End to end on 100 tuning drawings (`outputs/novelty/e2e_tuning_v1.md`, figure `e2e_tuning_v1_examples_0.png`)

Nothing about the omission is given. Optimistic only through the locator threshold (chosen on these drawings).

| omissions | pipeline | found | flagged as a region | fixed (strict) | recovered (2 px) | tol. mIoU in the region | violations per hole (truth) |
|---|---|---|---|---|---|---|---|
| medium + large (200) | shipped | 72% | 2% | 10% | 10% | 37.7 | 26.5 (9.6) |
| medium + large (200) | **hybrid v1** | **91%** | **76%** | **30%** | **42%** | **41.1** | **18.5** |
| large (100) | shipped | 77% | 1% | 7% | 9% | 36.5 | 31.0 (12.0) |
| large (100) | **hybrid v1** | 99% | 85% | **29%** | **42%** | 40.6 | 21.2 |
| small (100) | shipped | 67% | 2% | 52% | - | 63.0 | 10.3 (3.8) |
| small (100) | hybrid v1 | 68% | 10% | 50% | - | 62.3 | 10.5 |

Paired differences, hybrid minus shipped, medium + large: fixed +20 points [+13.5, +26], recovered +31 [+24.5, +38],
tolerant mIoU +3.4 [+1.6, +5.3]. Small omissions: fixed -2 [-5, 0] (inside the noise).
With the box given by hand (medium + large): recovered 40% (shipped B4) -> 54% (new fill), tolerant mIoU 38.9 -> 43.1.

Where the hybrid still loses cases (same file, broken down):
- missing whole rooms: flagged as a region in 89%, but only 44% of the flagged ones are recovered -> the FILL is
  the limit (the room's outline has to be guessed);
- erased rectangles: 63% of the flagged ones are recovered, but only 67% are flagged -> the LOCATOR is the limit.

Cost on intact drawings (100): a region flag on 16% of them (shipped: 1%); those drawings get 5.9 added components
each, so the mean rises from 0.92 to 1.57 added components per intact drawing.

Tried and rejected on these drawings (see `docs/decisions.md`): painting the locator's outline into the sketch
("outline prior": strict fixed +4.5 points, but recovered -13 and added-wall precision 0.70 -> 0.47), the
fill-consistency gate, and test-time augmentation of the locator.

### 4.4 Fill v2 (edit mode) against v1, same 100 tuning drawings (`outputs/novelty/e2e_tuning_v2.md`)

| medium + large (200 cases) | fixed (strict) | recovered (2 px) | added-wall precision | tol. mIoU in the region | violations per hole |
|---|---|---|---|---|---|
| hybrid v1, automatic | 30% | 42% | 0.70 | 41.1 | 18.5 |
| **hybrid v2, automatic** | 32% | **47%** | **0.81** | **45.5** | 19.8 |
| v1, box given | 42% | 54% | 0.71 | 43.1 | 21.0 |
| **v2, box given** | 42% | **62%** | **0.81** | **46.6** | 21.5 |

Paired v2 minus v1, automatic: recovered +5.5 points [+1.0, +10.0], tolerant mIoU +4.4 [+3.1, +5.9], added-wall
precision +0.09 [+0.06, +0.14], strict fixed +2.0 [-2.5, +6.5], violations +1.25 [+0.6, +1.9]. Small omissions and
intact drawings: no change (intact added components +0.06 [-0.10, +0.26]). By the rule fixed beforehand (the better
version on tuning drawings is the final one) **v2 is the final hybrid**. The expected side benefit - less damage
on a false region flag, because an edit model can keep what is there - did not show up.

## 5. Results on the held-out drawings (the reported numbers)

151 held-out val drawings, each with an intact version and one small, one medium and one large omission; nothing
about the omission is given to the automatic arms. Each hybrid version was frozen before its run and ran once:
v1 (inpaint fill) first, then v2 (edit fill) after it had been chosen over v1 on the TUNING drawings (section 4.4).
**v2 is the final hybrid** (`configs/novelty/pipeline_large_edit.yaml`); v1 is shown because it was also run.
Files: `outputs/novelty/e2e_heldout_final.md` (all arms, intervals, per kind; rows in `e2e_heldout_all_arms.csv`),
`e2e_heldout_v1.md`, `e2e_heldout_v2.md`; figures `e2e_heldout_v1_examples_0.png`, `e2e_heldout_v2_examples_0.png`
(the first six medium / large cases as they come, no picking). The shipped arms reproduce the separate baseline
run to the digit, so every arm saw the same erased drawings.

### 5.1 Automatic: nothing given

| omissions | pipeline | found | flagged as a region | fixed (strict) | recovered (2 px) | tol. mIoU in the region | violations per hole (truth) |
|---|---|---|---|---|---|---|---|
| large (151) | shipped | 80% | 3% | 13% [8, 19] | 11% [7, 17] | 38.6 | 29.6 (10.2) |
| large (151) | hybrid v1 | 95% | 80% | 32% [25, 40] | 42% [35, 50] | 42.3 | 21.4 |
| large (151) | **hybrid v2 (final)** | **95%** | **80%** | **36%** [29, 44] | **52%** [44, 59] | **45.3** | 22.5 |
| medium (151) | shipped | 73% | 1% | 15% [10, 21] | 17% [12, 24] | 42.6 | 19.9 (6.5) |
| medium (151) | **hybrid v2 (final)** | **87%** | **70%** | **38%** [30, 46] | **50%** [42, 58] | **47.3** | 15.3 |
| medium + large (302) | shipped | 76% | 2% | 14% [10, 18] | 14% [10, 18] | 40.6 | 24.7 (8.4) |
| medium + large (302) | hybrid v1 | 91% | 75% | 35% [30, 41] | 46% [41, 52] | 43.2 | 18.2 |
| medium + large (302) | **hybrid v2 (final)** | **91%** | **75%** | **37%** [32, 42] | **51%** [45, 57] | **46.3** | 18.9 |
| small (151) | shipped | 61% | 1% | 42% [34, 50] | - | 65.6 | 9.2 (3.4) |
| small (151) | hybrid v2 (final) | 64% | 8% | 42% [34, 50] | - | 65.5 | 9.2 |

Brackets: Wilson 95% intervals. Paired differences, final hybrid minus shipped (bootstrap over cases, 95%):

| omissions | fixed, points | recovered, points | tol. mIoU |
|---|---|---|---|
| large | +23.2 [+15.9, +30.5] | +41.1 [+32.5, +49.0] | +6.7 [+4.9, +8.5] |
| medium | +23.2 [+15.9, +31.1] | +33.1 [+25.2, +41.1] | +4.8 [+3.4, +6.2] |
| medium + large | +23.2 [+18.2, +28.5] | +37.1 [+31.5, +42.7] | +5.7 [+4.6, +6.9] |
| small | 0.0 [-2.0, +2.0] | - | 0.0 [-0.4, +0.3] |

On large omissions the automatic pipeline now fixes 2.8 times as many cases by the strict rule (13% -> 36%) and
4.7 times as many by the 2 px rule (11% -> 52%). Small omissions are handled exactly as before.

### 5.2 Which part helps (same cases, medium + large)

| arm | fixed | recovered (2 px) | added-wall precision | tol. mIoU |
|---|---|---|---|---|
| shipped pipeline | 14% | 14% | 0.75 | 40.6 |
| + locator, shipped B4 fill | 34% | 43% | 0.68 | 41.6 |
| + locator, fill v1 (retrained on parsed maps, inpaint) | 35% | 46% | 0.72 | 43.2 |
| + locator, fill v2 (edit mode) = final hybrid | 37% | 51% | 0.82 | 46.3 |
| box given by hand, shipped B4 fill | 41% | 46% | 0.66 | 42.7 |
| box given by hand, fill v1 | 43% | 59% | 0.74 | 45.6 |
| box given by hand, fill v2 | 45% | 66% | 0.84 | 48.4 |

- **The locator is the largest single step**: with the shipped fill it already takes "fixed" from 14% to 34%.
- **The new fill is a second, clear step**: final fill minus B4 with the same locator flags: recovered +7.9 points
  [+3.3, +12.6], tolerant mIoU +4.7 [+3.6, +5.8]; with the box given: recovered +19.5 [+13.6, +25.2], tolerant mIoU
  +5.7 [+4.5, +6.9]. Strict "fixed" does not separate the fills (+2.6 [-2.0, +7.3] / +3.6 [-2.0, +8.9]).
- **Edit mode over inpainting** (v2 minus v1, automatic): recovered +4.6 [+0.3, +9.3], tolerant mIoU +3.2 [+2.2, +4.2],
  added-wall precision 0.72 -> 0.82; on large omissions alone recovered +9.3 [+3.3, +15.2].
- The automatic final hybrid (51% recovered) is ahead of the shipped fill with the box given by hand (46%).
- **Fourier context blocks: a small effect.** Same recipe with and without the blocks, 120 tuning holes with the
  hole given (`outputs/novelty/tune_maps_ablation_*.md`): tolerant mIoU +1.0 to +1.3 points (interval excludes 0),
  wall recovery unchanged (+0.8 to +1.7 points, inside the noise). The backbone already has self-attention at its
  deepest level and one 256 px window covers most of a plan, so image-wide context was not what was missing.

### 5.3 By kind and on intact drawings

| omission kind | n | shipped, fixed | final hybrid, fixed | box given, fill v2, fixed |
|---|---|---|---|---|
| whole room | 115 | 11% | 35% | 38% |
| erased rectangle | 187 | 15% | 38% | 49% |
| small kinds (wall, window, doors, room walls, fixture, railing) | 151 | 42% | 42% | - |

| intact drawings (151) | flags per drawing | region flags per drawing | drawings with a region flag | added components per drawing |
|---|---|---|---|---|
| shipped | 0.64 | 0.01 | 1% | 0.68 |
| hybrid v1 | 0.74 | 0.15 | 13% | 1.38 |
| final hybrid (v2) | 0.74 | 0.15 | 13% | 1.69 |

Cost: a region flag on 13% of intact drawings (0.15 per drawing, inside the 0.2 budget set on tuning drawings),
and each false region flag redraws a whole area: added components per intact drawing rise from 0.68 to 1.69. On
erased drawings the region fill also draws more components that do not touch the omission (0.79 -> 2.28 per drawing).

### 5.4 Honest reading

- Large omissions went from "almost never repaired" (13% fixed, 11% recovered) to about one in three by the strict
  rule and one in two by the 2 px rule. Half are still not recovered.
- The limit differs by kind (tuning breakdown, section 4.3): a missing whole room is nearly always flagged but its
  outline must be guessed; an erased rectangle is filled well once flagged but is flagged only two times in three.
- The hybrid is more eager: one intact drawing in eight gets a region proposal it did not need, and v2 draws more
  there than v1 (1.69 vs 1.38 added components per intact drawing; on tuning drawings the two were equal).
  It suits a user who prefers a proposal to check over silence.
- Rule violations per hole fall against the shipped pipeline (24.7 -> 18.9) but the "valid" pass rate stays low
  (7% -> 11%): even the true layout pasted into a parser's map passes in only 2% of tuning cases.
- The Stage 2 test split has not been used. `python -m src.novelty.eval_e2e --split test --test-approved-by-user`
  is prepared for ONE run of the final hybrid if the user approves it.

## 6. Status and open items

- Done, round 1: large-omission set, locator, fill v1 and v2, Fourier ablation, tuning choices, one held-out run per
  version.
- Done, round 2 (section 8): bigger set with ink maps, wall-continuation rays, structure net (+ ablations without
  ink / without rays), wall-probability threshold, pipeline v3a / v3b, tuning choice, one held-out run of v3b.
- Not done: the room-graph idea of the report's section 8.2; the Gradio app still runs the shipped pipeline
  (`src/novelty/pipeline.run(image)` is the drop-in call for a hybrid tab).
- Open decision for the user: one test-split run of the final hybrid (prepared, guarded, not run).
- Where the next gain is (section 8.6): erased rectangles on faint drawings, the exact pixel position of the walls
  (strict rule), and drawing less around a region fill.

## 7. How to reproduce

| step | command / kernel | output |
|---|---|---|
| set | `kaggle/kernels/novelty/large_data` (`src/novelty/build_set.py`) | `large_train_*.zip`, `large_tune_*.zip` |
| locator | `kaggle/kernels/novelty/locate` | `runs/locator_s{0,1}/checkpoints/best.pt` |
| fill v1 / ablation | `kaggle/kernels/novelty/fill_ffc`, `fill_plain` | `runs/fill_{unet,diffusion}_{ffc,plain}/` |
| fill v2 (final) | `kaggle/kernels/novelty/fill_edit` (needs v1 and the locators) | `runs/fill_{unet,diffusion}_edit/` |
| tuning tables | `kaggle/kernels/novelty/eval` with `STAGE = "tune"` or `"e2e_tuning"` | `outputs/novelty/tune_*.md`, `e2e_tuning_*.md` |
| held-out tables | `kaggle/kernels/novelty/eval` with `STAGE = "e2e_heldout"`; `baseline` for the shipped arms | `outputs/novelty/e2e_heldout_*.md` |
| round 2: set | `kaggle/kernels/novelty/large_data2` (`build_set.py --versions 9 --ink`) | `large9_train_*.zip`, `large9_tune_*.zip` |
| round 2: structure net | `kaggle/kernels/novelty/structure` (ablations: `structure_abl`; stroke map: `structure2`) | `runs/struct_full_s{0,1}/checkpoints/best.pt` |
| round 2: tuning / held-out | `kaggle/kernels/novelty/eval` re-pointed with `point_eval.py` (`e2e_tuning`, `e2e_heldout`) | `outputs/novelty/e2e_tuning_v3{a,b}.md`, `e2e_heldout_v3b.md`, `e2e_heldout_round2.md` |
| local use | `from src.novelty.pipeline import run; result = run(image)` with the checkpoints at the paths in `configs/novelty/pipeline_structure_b.yaml` (the default) | same result dict as `run_pipeline` |

Kaggle GPU time: round 1 about 17 session hours, round 2 about 9 more in the same week (of 30) plus the last two runs, which ended after the weekly reset of 2026-11-14.

## 8. Round 2: "structure first" (the final hybrid)

Asked for after round 1: at least 60-70% end to end on large omissions, same dataset. Round 1 ended at 52%
recovered / 36% fixed. Round 2 keeps the round-1 fill (v2, edit mode) and changes how the missing part is found and
how walls are decided.

### 8.1 What the tuning drawings showed before any new training

| finding (423 erased tuning versions, map level) | number |
|---|---|
| The fill U-Net knows more walls than it draws: a wall whose position is uncertain by a few pixels never wins the per-pixel arg-max. Drawing a wall wherever its probability is >= 0.18, hole given | recovered 61.7% -> 73.5%, fixed 39.7% -> 58.2% |
| Same change with the locator's boxes instead of the true hole | recovered 48.2% -> 55.3% (finding the place is the gap) |
| Wider boxes (+16 px) or the whole plan as one hole | worse (48% -> 38%; 14-94% of intact drawings flagged) |
| Truly missing wall pixels that lie on the straight continuation of a still-visible wall end | 32% (rays are a hint, not a method) |

### 8.2 What was added

```
drawing -> PARSE -> CHECK (shipped) -> small flags -> shipped small-hole fill                          (unchanged)
        -> ink map
   parsed map + rays + ink -> STRUCTURE NET -> missing walls -> groups of walls = region flags
                                                             -> edit fill (v2) with the walls painted in -> diffusion finish
```

| part | what it is | file |
|---|---|---|
| Bigger set | Same training drawings, same generator and seeds, 9 erased versions per drawing instead of 3 (35,339 erased + 3,944 intact). Tuning and held-out cases are byte-identical to round 1. | `src/novelty/build_set.py --versions 9 --ink` |
| Ink map | For each map pixel, the share of drawing pixels darker than grey 128 under it. Tells "nothing is drawn here", which the parser's 16 classes cannot. | `src/novelty/data.py: ink_map` |
| Wall-continuation rays | From every end of a straight wall run, a ray through free space until the next wall; two channels (ends in free space / reaches a wall). | `src/novelty/rays.py` |
| Structure net | U-Net (ImageNet ResNet-34) on the WHOLE plan, no box: input parsed map + rays + ink; output "a missing wall runs here" and "this area is missing" (helper task). | `src/novelty/structure.py`, `configs/novelty/structure.yaml` |
| Decode | wall heat >= 0.5 -> centre line -> pieces shorter than 6 px dropped -> redrawn 3 px wide. | `structure.decode_walls` |
| Region flags | Groups of added walls (pieces within 8 px, at least 12 px, 6 px margin). The round-1 locator is no longer used for flags. | `structure.wall_clusters`, `pipeline.structure_holes` |
| Fill | Round-1 edit U-Net + diffusion in each flagged box, with three changes: walls where the U-Net's wall probability >= 0.18; the structure net's walls painted into the sketch; structure walls the diffusion dropped are put back, except over a door / window it drew. | `pipeline.hybrid_fill` (`fill.wall_tau`, `fill.keep_walls`) |
| Configuration | v3b (final): one structure net (seed 0), locator off. v3a: two nets averaged + locator. | `configs/novelty/pipeline_structure_b.yaml`, `pipeline_structure.yaml` |

### 8.3 Chosen on the tuning drawings

Structure net alone, map level, fully automatic (`outputs/novelty/tune_structure_sweep.txt`; 423 erased + 212 intact):

| structure net | recovered (2 px) | large | whole rooms | rectangles | recall / precision | intact drawings flagged |
|---|---|---|---|---|---|---|
| **all inputs, seed 0, threshold 0.5 (chosen)** | **73.5%** | 70.8% | 86% | 62% | 0.76 / 0.76 | 6% |
| all inputs, thresholds 0.2-0.7 | 73.5-74.0% | | | | | 6% |
| two seeds averaged, threshold 0.4 | 74.0% | 70.1% | 86% | 63% | 0.79 / 0.74 | 11% |
| **without the ink map** | **55%** | 53% | 66% | 45% | 0.59 / 0.71 | 20% |
| without the rays | 71-72% | | | | | |
| + round-1 locator boxes and fill walls | 74-75% | | | | | 21% |

End to end on 100 tuning drawings (`e2e_tuning_v3a.md`, `e2e_tuning_v3b.md`; same cases as sections 4.3 / 4.4):

| large omissions (100) | found | fixed (strict) | recovered (2 px) | small omissions fixed | intact with a region flag | added components / intact drawing |
|---|---|---|---|---|---|---|
| shipped | 77% | 7% | 9% | 52% | 1% | 0.92 |
| round 1 final (v2) | 99% | 30% | 47% | 50% | 16% | 1.63 |
| v3a (2 nets + locator) | 99% | 48% | 72% [63, 80] | 48% | 18% | 2.01 |
| **v3b (1 net, no locator) - chosen** | 98% | 46% | 70% [60, 78] | 51% | 6% | 1.27 |

v3b chosen: the same accuracy as v3a within the noise, a third of the false alarms, small omissions untouched.

### 8.4 Held-out result (the reported numbers; v3b frozen before the run, ran once)

Files: `outputs/novelty/e2e_heldout_round2.md` (all arms of both rounds, intervals, paired differences; rows in
`e2e_heldout_all_arms.csv`), `e2e_heldout_v3b.md`, figure `e2e_heldout_v3b_examples_0.png` (first six medium / large
cases as they come). Same 151 drawings and the same erased cases as section 5 (checked case by case).

| omissions | pipeline | found | flagged as a region | fixed (strict) | recovered (2 px) | recall / precision | tol. mIoU | violations per hole (truth) |
|---|---|---|---|---|---|---|---|---|
| large (151) | shipped | 80% | 3% | 13% [8, 19] | 11% [7, 17] | 0.13 / 0.70 | 38.6 | 29.6 (10.2) |
| large (151) | round 1 (v2) | 95% | 80% | 36% [29, 44] | 52% [44, 59] | 0.51 / 0.80 | 45.3 | 22.5 |
| large (151) | **round 2 (v3b)** | **96%** | **91%** | **48%** [40, 56] | **75%** [67, 81] | 0.79 / 0.74 | 44.0 | 22.0 |
| medium (151) | shipped | 73% | 1% | 15% [10, 21] | 17% [12, 24] | 0.21 / 0.79 | 42.6 | 19.9 (6.5) |
| medium (151) | round 1 (v2) | 87% | 70% | 38% [30, 46] | 50% [42, 58] | 0.52 / 0.83 | 47.3 | 15.3 |
| medium (151) | **round 2 (v3b)** | **91%** | **79%** | **54%** [46, 62] | **72%** [64, 78] | 0.75 / 0.79 | 45.9 | 14.4 |
| medium + large (302) | shipped | 76% | 2% | 14% [10, 18] | 14% [10, 18] | 0.17 / 0.75 | 40.6 | 24.7 (8.4) |
| medium + large (302) | round 1 (v2) | 91% | 75% | 37% [32, 42] | 51% [45, 57] | 0.52 / 0.82 | 46.3 | 18.9 |
| medium + large (302) | **round 2 (v3b)** | **93%** | **85%** | **51%** [45, 57] | **73%** [68, 78] | 0.77 / 0.77 | 45.0 | 18.2 |
| small (151) | shipped | 61% | 1% | 42% [34, 50] | - | | 65.6 | 9.2 (3.4) |
| small (151) | round 2 (v3b) | 62% | 5% | 42% [34, 50] | - | | 65.4 | 9.3 |

Paired differences (bootstrap over cases, 95%):

| large omissions | fixed, points | recovered, points | tol. mIoU |
|---|---|---|---|
| round 2 minus shipped | +35.1 [+27.2, +43.0] | +64.2 [+55.6, +72.2] | +5.4 [+3.7, +7.0] |
| round 2 minus round 1 | +11.9 [+5.3, +19.2] | +23.2 [+15.9, +31.1] | -1.3 [-2.6, +0.1] |
| medium + large: round 2 minus shipped | +37.4 [+32.1, +43.0] | +59.3 [+53.6, +64.6] | +4.4 [+3.1, +5.6] |
| medium + large: round 2 minus round 1 | +14.2 [+9.3, +19.2] | +22.2 [+16.6, +27.8] | -1.4 [-2.4, -0.3] |
| small: round 2 minus shipped | 0.0 [-2.0, +2.0] | - | -0.2 [-0.5, +0.1] |

By kind (held-out, automatic; recovered / fixed / flagged as a region):

| omission kind | n | shipped | round 1 (v2) | round 2 (v3b) | round 2, box given by hand |
|---|---|---|---|---|---|
| whole room | 115 | 8% / 11% / 3% | 52% / 35% / 86% | **80% / 52% / 94%** | 83% / 68% |
| erased rectangle | 187 | 18% / 15% / 2% | 50% / 38% / 68% | **69% / 50% / 80%** | 80% / 59% |

With the box given by hand (medium + large): recovered 46% (shipped B4) / 66% (round 1) -> 81% [77, 85]; fixed
41% / 45% -> 62% [57, 68]. Of the large omissions flagged as a region (138 of 151), 81% are recovered.

Intact drawings (151):

| pipeline | flags / drawing | region flags / drawing | drawings with a region flag | added components / drawing |
|---|---|---|---|---|
| shipped | 0.64 | 0.01 | 1% | 0.68 |
| round 1 (v2) | 0.74 | 0.15 | 13% | 1.69 |
| round 2 (v3b) | 0.70 | 0.08 | 7% | 1.02 |

### 8.5 The ink map: what it reads, and the check

- Most of the round-2 gain needs the ink map: without it the same net recovers 55% of the tuning cases (with it
  73.5%) and flags 20% of intact drawings (with it 6%).
- Trace check on the tuning drawings (does the erasure leave a ghost of the wall?): erased wall pixels carry ink on
  0.2% of their pixels, the blank area around them 0.8%, kept walls 47%; mean grey 248.2 vs 248.7. The map is
  binarised at grey 128, so paper tone and smudges are invisible. The net does not read leftover wall traces.
- What it does read: the BLANK AREA the omission leaves. The generators clear every stroke in the omitted room or
  rectangle (labels, furniture, dimensions too - the project's definition of a large omission since 2026-10-20), so
  "an area with no ink at all inside a building" is a strong cue. For an erased rectangle the edge where the
  neighbouring ink stops is straight; a region a person really forgot would have a softer edge. The numbers
  therefore hold for this kind of omission (a part of the plan not drawn at all), not for a room whose walls are
  missing but whose furniture and labels are present; that case is the no-ink row (55%).

### 8.6 Honest reading and what is left

- By the 2 px rule the target is met on held-out drawings: three large omissions in four are found and redrawn with
  nothing given (75% [67, 81]); by the strict pixel rule it is about one in two (48% [40, 56]). The strict rule tops
  out at 72% for these omissions (section 3), so both numbers have to be shown together.
- The strict rule is limited by exact wall position, not by wall thickness: redrawing the structure net's walls at
  5 px, at the predicted band width, or eroded gives 38-45% against 44% on the tuning maps (2026-11-14 check,
  decisions.md), so the 3 px decode stays.
- Whole rooms: 80% recovered. Erased rectangles: 69%; they are the weaker kind (flagged 80% of the time).
- Faint drawings (walls lighter than grey 128, 20% of the tuning cases) recover 55% against 82% for the rest: the
  ink map is nearly empty there. A stroke-map input (local contrast instead of a fixed grey level) was trained to
  address this; outcome in decisions.md.
- Region tolerant mIoU is slightly below round 1 (44.0 vs 45.3 on large; interval touches 0): round 2 draws more
  walls, and room types around them are not better.
- Costs: a region flag on 7% of intact drawings (round 1: 13%, shipped: 1%); on erased drawings 2.3 added
  components per drawing that do not touch the omission (shipped 0.8, round 1 2.3); 7 s per drawing on CPU for a
  region fill (shipped 3 s). "Valid" pass rate stays low (13%), as for the true layout pasted in (2%).
- The Stage 2 test split has still not been used.

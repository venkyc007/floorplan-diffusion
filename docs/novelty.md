# Novelty assessment: large omissions, end to end

Material for the novelty assessment (numbers, tables, figures, what each part does). It is not report prose.
Every number names its file in `outputs/novelty/`; every choice and its reason is in `docs/decisions.md`
("Novelty assessment"). Code: `src/novelty/` (its README lists the files). **Status: work in progress, see section 6.**

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
| Fourier context blocks | Residual FFT blocks (after LaMa) at the three deepest stages of both networks: every output pixel sees the whole window in one step. Zero-initialised, so the warm-started network starts exactly as B4. | `src/novelty/ffc.py` |
| Hybrid pipeline | The shipped pipeline with two switch-in points: region flags from LOCATE (they swallow the small flags inside them) and the new fill for region flags / large manual boxes. Small omissions take the shipped path untouched. | `src/novelty/pipeline.py`, `configs/novelty/pipeline_large.yaml` |

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

## 5. Results on the held-out drawings

Hybrid v1, frozen before this run (locator threshold 0.4, t0 700, N = 2; `configs/novelty/pipeline_large.yaml`).
151 held-out val drawings, each with an intact version and one small, one medium, one large omission; nothing about
the omission is given to the automatic arms. File: `outputs/novelty/e2e_heldout_v1.md` (per-case rows in the two
csv files; figure `e2e_heldout_v1_examples_0.png` = the first six medium / large cases as they come, no picking).
The shipped arms reproduce the separate baseline run exactly (42% / 15% / 13% fixed), so the two runs see the
same cases.

### 5.1 Automatic: nothing given

| omissions | pipeline | found | flagged as a region | fixed (strict) | recovered (2 px) | tol. mIoU in the region | violations per hole (truth) |
|---|---|---|---|---|---|---|---|
| large (151) | shipped | 80% | 3% | 13% [8, 19] | 11% [7, 17] | 38.6 | 29.6 (10.2) |
| large (151) | **hybrid v1** | **95%** | **80%** | **32%** [25, 40] | **42%** [35, 50] | **42.3** | **21.4** |
| medium (151) | shipped | 73% | 1% | 15% [10, 21] | 17% [12, 24] | 42.6 | 19.9 (6.5) |
| medium (151) | **hybrid v1** | **87%** | **70%** | **39%** [32, 47] | **50%** [42, 58] | 44.0 | **14.9** |
| medium + large (302) | shipped | 76% | 2% | 14% [10, 18] | 14% [10, 18] | 40.6 | 24.7 (8.4) |
| medium + large (302) | **hybrid v1** | **91%** | **75%** | **35%** [30, 41] | **46%** [41, 52] | **43.2** | **18.2** |
| small (151) | shipped | 61% | 1% | 42% [34, 50] | - | 65.6 | 9.2 (3.4) |
| small (151) | hybrid v1 | 64% | 8% | 42% [35, 50] | - | 64.8 | 9.1 |

Brackets: Wilson 95% intervals. Paired differences, hybrid minus shipped (bootstrap over cases, 95%):

| omissions | fixed, points | recovered, points | tol. mIoU |
|---|---|---|---|
| large | +19.2 [+12.6, +26.5] | +31.8 [+23.8, +39.1] | +3.6 [+1.5, +5.7] |
| medium | +24.5 [+16.6, +33.1] | +33.1 [+25.1, +41.7] | +1.5 [-0.6, +3.7] |
| medium + large | +21.9 [+16.9, +26.8] | +32.5 [+27.2, +38.1] | +2.6 [+1.2, +4.0] |
| small | +0.7 [-1.3, +2.6] | - | -0.7 [-1.4, -0.2] |

So on large omissions the automatic pipeline now fixes about 2.5 times as many cases by the strict rule (13% -> 32%)
and about 4 times as many by the 2 px rule (11% -> 42%), and small omissions are handled as before.

### 5.2 Which part helps (ablation on the same cases, medium + large)

| arm | fixed | recovered (2 px) | added-wall precision | tol. mIoU |
|---|---|---|---|---|
| shipped pipeline | 14% | 14% | 0.75 | 40.6 |
| + locator, shipped B4 fill | 34% | 43% | 0.68 | 41.6 |
| + locator, new fill (hybrid v1) | 35% | 46% | 0.72 | 43.2 |
| box given by hand, shipped B4 fill | 41% | 46% | 0.66 | 42.7 |
| box given by hand, new fill | 43% | 59% | 0.74 | 45.6 |

- **The locator is most of the gain**: giving the shipped fill the locator's region flags already takes "fixed" from
  14% to 34%.
- **The new fill adds a smaller, real part**: new fill minus B4 with the same locator flags: recovered +3.3 points
  [0.0, +6.6], tolerant mIoU +1.5 [+0.6, +2.4]; with the box given: recovered +12.6 [+7.9, +17.2], tolerant mIoU +3.0
  [+1.9, +4.0]. Strict "fixed" does not separate the two fills (+1.3 [-1.3, +4.3] / +2.0 [-2.0, +6.3]).
- The automatic hybrid (46% recovered) now equals the shipped fill with the box given by hand (46%).

### 5.3 By kind and on intact drawings

| omission kind | n | shipped, fixed | hybrid v1, fixed | box given, new fill |
|---|---|---|---|---|
| whole room | 115 | 11% | 30% | 33% |
| erased rectangle | 187 | 15% | 39% | 49% |
| small kinds (wall, window, doors, room walls, fixture, railing) | 151 | 42% | 42% | - |

Cost: a region flag on 13% of the intact drawings (shipped: 1%; 0.15 region flags per drawing, inside the 0.2
budget set on tuning drawings). Added components per intact drawing rise from 0.68 to 1.38, and on erased drawings
the region fill draws more components that do not touch the omission (0.79 -> 1.99 per drawing).

### 5.4 Honest reading

- Large omissions went from "almost never" (13% fixed) to about one in three by the strict rule and four in ten by
  the 2 px rule. Most are still not repaired.
- The limit differs by kind (tuning breakdown, section 4.3): a missing whole room is nearly always flagged but its
  outline must be guessed; an erased rectangle is filled well once flagged but is flagged only two times in three.
- A third of the cost of a false region flag is paid by intact drawings; the hybrid is for a user who prefers a
  proposal to check over silence.
- Tolerant mIoU in the omission region moves by only +2.6 points: it compares every class of the box with ONE
  ground truth, and a redrawn room rarely has the same room types and fixtures as the erased one.

## 6. Status and open items

_Updated in `CLAUDE.md` ("Novelty assessment")._

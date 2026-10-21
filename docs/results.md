# Results: what the final system does and how well

All numbers come from files in `outputs/` (named in each table). **Stage 1 has test numbers** (evaluated once with
settings frozen on val). **Stage 2 numbers are all on validation** until the one-shot test run
(`docs/final_test_notebook.md`); where a number was measured on the drawings a setting was tuned on, it says so,
because those numbers are optimistic. Superseded results are listed at the end and kept in `outputs/stage2/history/`.

## 1. The final system in one page

### Stage 1 — mask-given inpainting (ResPlan, 128 × 128, 6 classes)

1. **Input**: a plan with a region blanked out + the mask (the removed part's bounding box with 2–6 px random padding,
   so the mask never reveals the shape).
2. **Sketch**: the matched U-Net (diffusers UNet2DModel, 25.3M parameters, weighted cross-entropy, 30k steps × 64)
   draws the hole in one pass.
3. **Re-noise**: the sketch is noised to t₀ = 400 / 600 / 700 for small / medium / large holes (mask < 2.7% /
   < 7.7% / larger of the building).
4. **Finish**: a Palette-style diffusion model (x0-prediction, cosine schedule, 1,000 steps), fine-tuned 26k steps
   with two structural losses (soft clDice 0.01 + room closure 0.03), samples with stochastic DDIM from t₀, resetting
   the known region at every step.

### Stage 2 — complete a real drawing (CubiCasa5K, 16 classes): PARSE → CHECK → FILL → VERIFY

| step | what happens | model / setting (`configs/stage2/pipeline.yaml`, `detector_threshold.yaml`) |
|---|---|---|
| PARSE | the drawing becomes a 16-class map | parser v2B: U-Net with an ImageNet ResNet-34 encoder, CE + Dice, then fine-tuned with 8 constraint losses and room names erased on half the crops; parsed at the scale that gives ~8 px walls, rescaled to ~3 px walls |
| CHECK | find what is missing | union of: learned detector v2 (parsed map + violation maps) and v3 (drawing + parse), rules C8 (room cut off → door missing), R5 (doorless gap between rooms that need a door), door-ink and window-ink checks (parsed symbol with no ink in the drawing), R6 (windowless exterior wall), R8 (merged room, fixture-mix test only); budget 0.4 false alarms per intact tuning drawing |
| FILL | draw each flagged hole | small holes (≤ 100 px): 16-class U-Net sketch + the flagged component painted in → re-noised to t₀ = 300 → constrained 16-class diffusion → keep only the flagged type. Large holes / big manual boxes: B4 models (fine-tuned on 256 px windows with 50% large holes) on one big window |
| VERIFY | pick the best sample | best-of-N (N = 2 live, 4 evaluation) by the fewest structural rule violations; x0 constraint guidance exists but is off (no measurable effect) |

Output: the additions drawn in red on the user's drawing (doors as gap + leaf + arc) and a list such as
"bedroom had no door → door added on the bedroom / hallway wall".

## 2. What it captures

- **16 classes** (train pixel share, `outputs/stage2/data_stats.md`): wall 6.9%, door 0.81%, window 1.51%, railing 0.46%;
  living 7.9%, bedroom 8.0%, kitchen 3.8%, bathroom 1.8%, hallway 3.4%, other room 11.0%; outdoor 5.8%, background
  47.6%; toilet 0.15%, sink 0.34%, bath/shower 0.08%, appliance 0.40%.
- **Omission kinds CHECK looks for** (simulated by erasing on the drawing): door drawn over with wall (a), door leaf
  + arc erased (b), window, wall piece, railing, fixture, a room's walls, and the large kinds "whole room" and
  "region".
- **Found per kind** (a flag overlaps the omission; in brackets: the flag also has the right type):

| kind | TUNING drawings, final setting (optimistic) `check_tune.md` | HELD-OUT drawings, nearest setting `detection_eval.md` |
|---|---|---|
| false alarms per intact drawing | 0.30 | 0.59 |
| mean recall | 71% | 60% |
| door (a) | 72% | 72% (66%) |
| door (b) | 72% | 41% (34%) |
| window | 89% | 66% (66%) |
| wall | 83% | 76% (41%) |
| railing | 0% | 11% (0%) → manual mode only |
| fixture | 58% | 66% (39%) |
| room (its walls) | 100% | 89% (71%) |
| whole room / region | 89% / 76% (but no region flag: these get a small fill only) | not in the held-out set |

The exact final CHECK has no held-out number yet (the held-out row differs in detector v2 threshold, R6 scale and
R8); the test run gives the first unbiased number.

## 3. Step-by-step numbers

### PARSE (159 held-out val drawings, `outputs/stage2/parser_v2.md`, %)

| input | 16-class mIoU | wall IoU | door IoU | window IoU | door F1 | window F1 | room-type mIoU |
|---|---|---|---|---|---|---|---|
| **v2B, normal drawing** | **70.1** | 77.6 | 66.6 | 75.8 | **79.5** | **77.3** | **76.3** |
| v2B, room names erased | 66.2 | 77.9 | 65.3 | 76.1 | 78.3 | 77.3 | 64.3 |
| v1, normal | 70.7 | 77.4 | 65.6 | 74.6 | 78.6 | 74.5 | 78.1 |
| v1, names erased | 61.6 | 77.1 | 62.9 | 73.8 | 75.6 | 74.9 | 54.9 |

v2B was chosen because it depends far less on the room names (room-type mIoU 64.3 vs 54.9 with names erased).
Note: v2B was chosen from this table, so it is not fully "held-out" for v2B.

### CHECK

| setting | false alarms / intact drawing | mean recall | file |
|---|---|---|---|
| final, tuning drawings | 0.30 | 71% | `check_tune.md` |
| detector v3 alone + rules, tuning | 0.28 | 68% | `check_tune.md` |
| rules + ink checks only, tuning | 0.23 | 25% | `check_tune.md` |
| nearest-to-final, held-out | 0.59 | 60% | `detection_eval.md` (last section) |
| same without the window-ink check, held-out | 0.32 | 58% | `detection_eval.md` |

Fake-erasure control (`fake_erasure.md`): detector v3 fires on 23% of erased furniture, 2% of blank patches (2% on
untouched ones) and 0% of erased words — partly it reacts to erasure traces, not only to missing architecture.

### FILL with the hole given

| case | method | fixed (IoU ≥ 0.3) | tolerant mIoU | validity | file |
|---|---|---|---|---|---|
| 30 small val holes | **K2a U-Net + K1 constrained diffusion (final for small)** | 25/30 | **69.2** | 63% | `b4_val.md` |
| 30 small val holes | B4 models | 25/30 | 68.5 | 67% | `b4_val.md` |
| 30 large val holes (256 px) | **B4 + one big window (final for large)** | **29/30** | **49.3** | **47%** | `b4_val.md` |
| 30 large val holes | current models, big window / tiled | 26/30 / 22/30 | 41.9 / 38.1 | 30% / 37% | `b4_val.md` |
| 20 held-out drawings, medium + large, box given | B4 + big window | 6/20 | 27.9 → 32.1 | 0% → 15% | `large_fill_compare_v2.md` |

Per component type (150 tuning holes, `eta_tune.md`): door 83.5, window 85.6, wall 76.6, railing 69.9, fixture 67.8,
room 46.5 tolerant mIoU.

### VERIFY and sampling

| question | answer | file |
|---|---|---|
| does the diffusion finish beat the U-Net sketch? | yes on structure: violations per hole 3.70 vs 10.70 (ground truth 3.53), validity 77% vs 27%, tolerant mIoU 69.2 vs 67.1 (30 holes) | `t0_sweep.md` |
| does x0 constraint guidance help? | no: eta 0.1–3 give identical output; only absurd strengths (≥ 3·10⁵) change anything, and they hurt | `eta_tune.md` |
| constrained vs control diffusion (Stage 2) | in-training val only: hole mIoU 0.397 vs 0.372, same violation sums; the paired test is in the final test | `outputs/kaggle/k1/.../val_log.csv` |
| constrained vs control (Stage 1) | no measurable difference (0.751 vs 0.751; 0.801 vs 0.801 on full val) | `outputs/stage1/finetune_ablation.md`, `selection.md` |

### END TO END (fast settings, one omission per drawing)

| run | n | found | fixed (by the fill itself) | false additions | tolerant mIoU before → after | file |
|---|---|---|---|---|---|---|
| batch 2, 12 fresh held-out drawings, final config | 12 small | 9 | 8 (2) | 5 on 4 drawings | 76.5 → 74.4 | `e2e_val_batch2_b4.md` |
| batch 1 (v4) | 11 small + 1 large | 11 | 9 (3) | 3 on 1 drawing | 56.8 → 60.9 | `e2e_val_v4.md` |
| batch 3 (large kinds, before B4) | 4 whole room + 4 region + 4 small | 7 | 2 (1); large 1/7 | 10 | 44.8 → 44.9 | `e2e_val_batch3.md` |
| 20 large omissions, automatic | 7 medium + 13 large | 15 | 2 | — | 27.9 → 29.2 | `large_fill_compare_v2_auto.md` |
| re-check after the 2026-10-21 audit fixes (same drawings) | batches 1 / 2 / 3 | 11 / 9 / 6 | 9 / 8 / 2 | 3 / 5 / 10 | 56.8 → 61.6 / 76.5 → 74.4 / 44.8 → 44.9 | `e2e_val_v5.md`, `e2e_val_batch2_v2.md`, `e2e_val_batch3_v2.md` |

"Fixed" counts an omission when it was flagged and the final plan has the right class there (IoU ≥ 0.3), whoever
drew it; "by the fill itself" counts only the fill's own additions — the rest are symbols the parser redrew, which
CHECK then reports as "symbol missing in the drawing".

### Stage 1 (val selection / test once; `outputs/stage1/selection.md`, `stage1_test.md`)

| method | tolerant mIoU val / test | validity val / test | connectivity error val / test |
|---|---|---|---|
| **size-adaptive hybrid** | 0.842 / **0.846** | 88.0% / **88.0%** | 0.011 / **0.011** |
| matched U-Net | 0.846 / 0.850 | 80.4% / 81.0% | 0.092 / 0.085 |
| hybrid t₀ = 600 | 0.842 / 0.846 | 87.4% / 87.6% | 0.023 / 0.022 |
| diffusion refine-600 | 0.801 / 0.804 | 83.6% / 84.1% | 0.039 / 0.037 |
| nearest fill | 0.489 / 0.493 | 36.0% / 36.6% | 0.066 / 0.064 |

By hole size on test (hybrid): small 0.931 / 97.5%, medium 0.907 / 96.1%, large 0.700 / 70.7%. By type: door 0.948,
window 0.898, wall 0.924, region 0.784, room 0.675. Every metric of every method is within 2 points of val.

## 4. Honest weak spots

1. **Stage 2 has no test numbers yet**, and several val numbers are on the drawings the settings were tuned on.
2. **CHECK false alarms do not generalise**: 0.30 per intact drawing on tuning, 0.59 on held-out (budget 0.4).
3. **Right type is much rarer than "found"** on held-out drawings: wall 41%, door (b) 34%, fixture 39%.
4. **Large omissions are not detected automatically** (R8 size test does not fit the budget); they are fixed only when
   the user marks them (manual mode), and even then 6/20 on real drawings.
5. **"Fixed" is mostly the parser**: by addition only 2–3 of 12; the parser redraws erased doors (b) 80% of the
   time, windows 42%.
6. **The fill can lower local scores**: batch 2 tolerant mIoU 76.5 → 74.4; room holes score 46.5 even with the hole
   given.
7. **Constraint machinery has small or unproven effects**: guidance does nothing; Stage 1 constrained = control;
   Stage 2 +2.5 points in-training only.
8. **Room types depend on the printed room names** (76.3 → 64.3 when erased); R5 relies on room types.
9. **Synthetic omissions**: the evaluation erases parts of real drawings; real drafting omissions may look different
   (detector v3 partly reacts to erasure traces).
10. **Small samples**: 12 drawings per end-to-end batch, 30 holes per fill comparison.
11. **Scale mismatch found in the final audit**: the rasteriser draws polygons 1 px thicker (walls 4 px / 9 px instead
    of 3 / 8) and the pipeline's wall-thickness estimate reads 9.55 px, so Stage 2 maps are ~0.84× the scale the
    fill models were trained at. The detectors were trained at the pipeline's scale, so a consistent fix needs them
    retrained; documented, not changed.
12. **Stage 1 validity rewards leaving the hole empty** (a background fill scores 46% validity on 400 val cases): it
    measures "no wrong structure", so it is always read together with tolerant mIoU and F1.

## 5. Superseded results (history)

- Stage 1 path: U-Net baseline 0.775 → diffusion run 1 0.698 (DDIM 20) → refine-600 0.801 → matched U-Net 0.846 →
  hybrids → size-adaptive hybrid.
- CHECK on held-out drawings: 12–13% recall at 0.85–0.95 false alarms (18/19 Oct) → detectors v2 + v3 59% at 0.41
  (19 Oct) → 60% at 0.59 (20 Oct, nearest to final).
- Check set (12 drawings): 1/12 correct → 3/12 → 6/12.
- End-to-end batch 1: fixed 3/12 → 8/12 → 9/12.
- Sampler: eta 0.3 → 0.1 → 0 (off); t₀ by size bins → fixed 300 for small holes.

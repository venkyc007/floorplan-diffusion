# Overnight report (2026-10-18/19)

All evaluations ran as Kaggle CPU kernels (`floorplan-stage2-eval-{doors,checkset,text}`); the laptop only ran unit
tests and smoke tests on <= 3 items. No GPU jobs. Everything is committed locally on `stage2` (not pushed). Every
own choice is in `docs/decisions.md` ("Overnight 2026-10-18/19").

**Headline: on real drawings, CHECK finds only ~12% of omissions, mainly because the parser redraws erased doors and
windows from context, so they never look missing. The parser also reads the room names: without text, room-type
mIoU falls from 81 to 51.** Both need your decision (end of this file).

## Step 1: realistic erasure

- `src/stage2/data/erasure.py`, used by the check set, the examples and the image-level evaluation. The component's
  label pixels (grown 2 px) are filled with cv2.inpaint (Telea, radius 3). Never white, never a box.
- Two own fixes after looking at the crops. Nearby walls / openings / fixtures are masked while inpainting and put back,
  otherwise a fixture against a wall got a dark smear. The 2 px growth never eats into other walls.
- **Door kind (a)**: the wall is drawn across the doorway as a "bridge" (the wall mask closed across the gap). It is
  painted line by line with that line's own wall grey, so outline-style walls get their two thin lines continued and
  solid walls get their fill. The leaf and arc are inpainted away. **Door kind (b)**: doorway + leaf + arc inpainted.
- The orange outline is drawn on the figure axes only; `tests/stage2/test_erasure.py` checks that the image given to
  the pipeline is single-channel and unchanged away from the spot (3 tests, pass).
- Look at: `outputs/stage2/erasure_check.png` (6 crops: 2 grey-shaded, 2 white, 2 door kind a).
- Known leftovers: tile texture becomes smooth in the spot; double-swing doors drawn as a box keep some strokes; doors in
  angled walls fall back to a plain fill; nearby text can lose a letter.
- **Failed once, fixed and re-run.** The first version painted door (a) in the "median wall colour". On outline-style
  walls that colour is light, so the doorway still looked open. The re-run uses the bridge above.

## Step 2: doorless-gap rule R5, final

- Fires on a straight break in a wall (length within the door range, wall at both ends) between two spaces that
  must meet these conditions:
  - the pair's direct-contact rate is < 10%;
  - one side is a bedroom, a bathroom, or an other_room >= p25 bedroom area (1514 px);
  - the other side is not an other_room < p25 other_room area (460 px, closets).
- Cut-offs were chosen on the tuning plans only (`detect/gap_tune.py`, `outputs/stage2/gap_rule.md`).
- **Needs your OK: how I read the budget.** The tuning split's detector alone is already at 0.50 false alarms per
  intact plan, so "total <= 0.4" can't hold on tuning plans. I let R5 add at most 0.4 - 0.31 (detector + C8,
  held-out) = 0.09 per intact plan. To fit, the shortest firing gap was raised to 11 px.
- Result on the tuning plans: 0.09 false alarms, door (b) recall 33% (6 of 18). The pair rule alone would give 72% at
  0.23 false alarms.
- Limitation (logged): on a drawing, R5 is only as good as the parser's room types, and those depend on text (step 5).
- Tests: `tests/stage2/test_gaps.py` (3 pass).

## Step 3: held-out detection on drawings (image level)

- Setup: the 159 held-out val plans. 59 intact drawings measure false alarms; the other 100 drawings get 200 omissions
  (door (a) and (b) 29 each, window 29, wall 29, railing 18, fixture 38, room 28), made with the step 1 erasure.
  Then the pipeline's PARSE + CHECK run (no fill).

| method | false alarms / intact drawing | mean recall | door (a) | door (b) | window | wall | railing | fixture | room |
|---|---|---|---|---|---|---|---|---|---|
| detector + C8 | 0.85 | 10% | 7% | 7% | 10% | 0% | 11% | 13% | 21% |
| detector + C8 + R5 (final) | 0.95 | 12% | 7% | 7% | 10% | 3% | 11% | 13% | 32% |

- Door (a) where the closure cuts a room off (C8's scope): 7% of 15. Door (b) in R5's scope: 0% of 14.
- **Why so low: the parser draws the removed component back.** Share of the removed pixels the parser still labels
  with the removed class:

  | door (a) | door (b) | window | wall | room walls | railing | fixture |
  |---|---|---|---|---|---|---|
  | 55% | 85% | 77% | 26% | 23% | 17% | 8% |

  For doors and windows, CHECK has nothing to find. Fixtures are not redrawn, but the detector, trained on ground-truth
  label maps, still misses them on parsed maps (domain gap). It also raises false railing / appliance holes:
  0.95 false alarms per intact drawing, against 0.31 on label maps.
- Compare: the label-level held-out numbers (same plans, ground-truth maps) were 62% mean recall at 0.31 false alarms.
  They stay in `outputs/stage2/detection_eval.md`; the new section is at the end. Per case: `outputs/stage2/image_eval_cases.csv`.

## Step 4: 12-drawing check set

- Setup: fast mode, t0 = 300, N = 2, type-restricted fill, C8 per building part, R5, step 1 erasure.
- **Score 1 / 12 correct, 2 partly.**
  - The intact drawing came back unchanged.
  - Partly: door (a) found by C8 at the right wall but nothing added; R5 fired at the open doorway but nothing added.
  - The rest are wrong: doors and windows redrawn by the parser; detector misses; false railing / appliance / R5 holes,
    with a railing or sink added in 4 drawings.
- Look at: `outputs/stage2/check_set.png`, plus `outputs/stage2/check_set.md` (per drawing: removed / detected / added,
  and my verdict with the reason).
- No setting was changed based on this set.
- Run twice: the first run (older door erasure) scored 2/12. Its case 2 got a door added; the re-run found the same
  hole but added nothing.

## Step 5: room-type text dependence (parser v1, 100 val drawings)

| room-type IoU | living | bedroom | kitchen | bathroom | hallway | other_room | mean |
|---|---|---|---|---|---|---|---|
| original | 89.2 | 89.4 | 80.0 | 77.3 | 77.1 | 73.5 | 81.1 |
| text erased | 64.3 | 61.1 | 73.9 | 36.8 | 23.5 | 43.9 | 50.6 |

- **-30.5 points.** Control: wall / door / window IoU 80.5 / 67.6 / 75.5 before vs 80.5 / 65.6 / 74.9 after, so
  only the text was removed.
- Without names, rooms collapse into other_room (hallway: 49% of its pixels go to other_room; bathroom: 38%).
- Look at: `outputs/stage2/text_erasure_examples.png` (4 before/after crops) and `outputs/stage2/text_dependence.md`
  (confusion matrices).
- **Failed once, fixed and re-run.** The first text mask also removed thin non-text lines on a small-scale drawing and
  left letter ghosts (it gave -24 points). Now only pieces inside rooms are removed, with a lighter ink threshold.
  Some partial words that cross a room boundary remain, so -30 is, if anything, an underestimate.
- **Proposal (not launched)**: parser v2 = fine-tune v1 with random text erasure (this mask + inpainting on ~50% of
  training drawings) plus the local constraint losses. Select on room-type mIoU of text-erased val drawings.

## Needs your decision

1. **The parser redraws missing doors and windows**, which hides them from CHECK (the biggest problem). Options:
   - (a) a cheap CPU check, no training: for each parsed door / window, look for its ink in the drawing (door arc,
     window lines). If the ink isn't there, report the component as missing and draw it in red.
   - (b) parser v2 trained with omission augmentation: erased components labelled as what is actually drawn, e.g. the
     wall or gap. Needs GPU.
   - (c) report this honestly as a limitation.

   My suggestion: (a) now, (b) folded into the parser v2 run if GPU time allows.
2. **The detector's domain gap**: retrain it on parser outputs of the train drawings (GPU), or accept and report it.
3. **Parser v2 for text dependence** (proposal above), possibly the same run as 1(b).
4. **R5 budget reading** (step 2): OK?
5. **Housekeeping, not deleted (your rule)**: untracked smoke-test files in `outputs/stage2/`: `check_set_smoke.*`,
   `image_eval_smoke.*`, `t0_sweep_preds.npz` (the t0-sweep cache). Delete them or keep?
6. The full-mode t0 = 300 row of the t0 sweep is still unfinished (13/30 holes before the laptop ran out of memory).
   I can run it as a CPU kernel if you want it.

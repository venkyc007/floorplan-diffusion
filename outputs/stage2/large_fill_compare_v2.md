# Large holes v2: the three fills with the B4 models (256 px large-hole fine-tune)

Same 20 held-out drawings as v1 (batch 3's 8 large-kind cases + 6 new medium + 6 new large; 7 medium, 13 large).
All large-hole strategies now use the B4 U-Net + diffusion (fill.large_models = b4). Kaggle CPU kernel
floorplan-stage2-b4-val v1, fast sampling, eta 0. Per-case tables: large_fill_compare_v2_manual.md / _auto.md;
figure: large_fill_compare_v2.png (manual mode, the first 2 cases of each group; input | B1 | B2 (t0 300, the better
manual mean) | B3 | ground truth; red = the hole box). Validation on val windows: b4_val.md.

## Validation with the hole given (val windows, label level; b4_val.md)

| holes | arm | fixed | tol. mIoU | valid |
|---|---|---|---|---|
| 30 large | current models, B1 big window | 26 / 30 | 41.9 | 30% |
| 30 large | current models, B3 tiled | 22 / 30 | 38.1 | 37% |
| 30 large | **B4, B1 big window** | **29 / 30** | **49.3** | **47%** |
| 30 small | current models | 25 / 30 | 69.2 | 63% |
| 30 small | B4 | 25 / 30 | 68.5 | 67% |

Routing (user rule): large / region holes -> B4; small holes stay on the current models (B4 is 0.7 points lower in
tolerant mIoU there, though 4 points higher in validity).

## The 20 drawings: v1 (current models) -> v2 (B4)

| strategy | fixed v1 -> v2 (manual) | tol. mIoU in the hole v1 -> v2 | valid v1 -> v2 | s per hole |
|---|---|---|---|---|
| B1 big window | 5 -> 6 / 20 | 34.5 -> 32.1 | 15% -> 15% | 22 |
| B2 coarse-to-fine, t0 150 | 3 -> 3 / 20 | 31.7 -> 31.5 | 0% -> 5% | 44 |
| B2 coarse-to-fine, t0 300 | 3 -> 3 / 20 | 31.7 -> 31.7 | 5% -> 5% | 44 |
| B3 tiled | 6 -> 5 / 20 | 33.9 -> 33.3 | 15% -> 15% | 19 |

(parse before the fill: tolerant mIoU 27.9, 0% valid). By size bin (B1, v2): medium 0 / 7 fixed, 30.5; large 6 / 13,
32.9.

Auto mode (full pipeline, CHECK with R5 on and R8 = fixture mix): found 15 / 20, **region flag 0 / 20** (R8 v2's size
test does not fit the budget, see decisions.md), so the large fills never run; fixed 2 / 20 for every strategy,
tolerant mIoU 27.9 -> 29.2, 0% valid, 20 s per drawing.

## Verdicts (figure cases, manual mode, B4 models)

| # | case | verdict |
|---|---|---|
| 1 | 3691 room (whole), large | B1 and B3 put the dividing walls back (IoU 0.37 / 0.44: fixed) and draw a hall and two rooms, close to the truth; B2 adds a stray wall piece and shifts the left wall |
| 2 | 14935 room (whole), large | all three draw one bathroom-like room with a window-like strip on the outer wall; the hall / bathroom wall is still missing (IoU <= 0.10) |
| 3 | 4773 region, medium | no change from v1: the two big rooms stay, the small partitions and the kitchen corner are not drawn (IoU 0.13-0.15) |
| 4 | 4864 room (whole), medium | the bathroom is still not recreated; B1 / B3 split the area between bedroom and living as in v1 (IoU 0.11-0.12) |
| 5 | 14616 room (whole), large | B2 now draws a closed room with a wall below it (IoU 0.18, nearest the truth of the three); B1 / B3 still mix kitchen and bedroom patches |
| 6 | 5080 room (whole), large | all three still cut the kitchen-living with a wall that is not there and add a bedroom (tol. mIoU 6-14) |

## Conclusion

B4 is clearly better than the current models on large holes in the setting it was trained for (val windows, label
level: +7 points of tolerant mIoU, +17 points of validity, 29 / 30 fixed), but on the 20 real drawings it changes
little: 6 / 20 fixed with B1 (5 before), tolerant mIoU 32-33 vs 34 before. On drawings the hole box is placed on the
parser's map: the parse around the hole has its own errors (merged rooms, missing doors), and the box includes outer
walls, so the context B4 learned from is noisier than in training. Setting (own choice, from the val result, which
is the selection set): fill.large_models = b4, fill.large_strategy = big (B1: the strategy B4 was validated with;
most fixed on the drawings). The open gap is detection: with R8 = fixture mix only, no large omission is flagged
automatically, so large holes are only filled in manual mode (the user draws the box).

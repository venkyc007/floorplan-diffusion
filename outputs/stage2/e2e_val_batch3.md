# End-to-end validation on 12 new held-out val drawings_batch3

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall break at a door flag (src/stage2/detect/flag_gap.py).

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | room (whole) (room) | exterior | edge | large (9.072%) | window (WINK+detector); room (detector); room (INK+detector) | yes | yes | no (0.15) | no (0.08) | 0 | - | 54.5 -> 55.3 | no -> no | 45 |
| 2 | colorful-14935 | room (whole) (room) | interior | centre | large (7.86%) | sink (detector) | yes | no | no (0.03) | no (0.00) | 0 | - | 37.8 -> 37.4 | no -> no | 19 |
| 3 | high_quality_architectural-8154 | room (whole) (room) | interior | edge | medium (6.441%) | none | no | no | no (0.00) | no (0.00) | 0 | - | 38.8 -> 38.8 | no -> no | 1 |
| 4 | colorful-6226 | room (whole) (room) | exterior | edge | large (14.473%) | room (detector) | yes | yes | yes (0.46) | yes (0.32) | 0 | - | 47.8 -> 49.3 | no -> no | 21 |
| 5 | high_quality_architectural-1376 | region (room) | interior | centre | large (24.718%) | door (R5); room (detector) | yes | yes | no (0.03) | no (0.03) | 1 | - | 8.2 -> 8.6 | no -> no | 33 |
| 6 | colorful-14938 | region (room) | exterior | centre | large (11.17%) | window (detector); sink (detector) | yes | no | no (0.02) | no (0.00) | 1 | - | 32.6 -> 32.6 | no -> no | 32 |
| 7 | colorful-6628 | region (room) | exterior | centre | large (13.917%) | door (INK) | no | no | no (0.00) | no (0.00) | 1 | - | 33.6 -> 33.6 | no -> no | 2 |
| 8 | high_quality-9975 | region (room) | exterior | centre | large (19.228%) | door (INK) | yes | no | no (0.01) | no (0.00) | 1 | - | 36.8 -> 36.8 | no -> no | 1 |
| 9 | high_quality_architectural-1654 | wall (wall) | interior | centre | small (0.313%) | sink (detector) | no | no | no (0.19) | no (0.00) | 1 | - | 50.5 -> 50.5 | no -> no | 16 |
| 10 | high_quality_architectural-1916 | room (room) | interior | centre | small (0.302%) | window (WINK) | no | no | no (0.00) | no (0.00) | 1 | - | 62.3 -> 62.3 | no -> no | 2 |
| 11 | high_quality_architectural-1067 | window (window) | exterior | edge | small (0.82%) | door (INK); window (detector) | yes | yes | yes (0.74) | no (0.00) | 1 | - | 95.3 -> 95.3 | no -> no | 1 |
| 12 | high_quality_architectural-1359 | door (b) (door) | exterior | centre | small (0.297%) | window (WINK); window (WINK); window (WINK) | no | no | no (0.54) | no (0.00) | 3 | - | 38.9 -> 38.9 | no -> no | 1 |

**Total:** found 7 / 12, right type 4 / 12, fixed 2 / 12 (by addition 1 / 12), false additions 10 (on 8 drawings); tolerant mIoU 44.8 -> 44.9; validity 0% -> 0%; time 15 s per drawing (max 45 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (b) | 1 | 0 | 0 | 3 | 38.9 |
| window | 1 | 1 | 1 | 1 | 95.3 |
| wall | 1 | 0 | 0 | 1 | 50.5 |
| room | 1 | 0 | 0 | 1 | 62.3 |
| room (whole) | 4 | 3 | 1 | 0 | 45.2 |
| region | 4 | 3 | 0 | 4 | 27.9 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 5 | 2 | 0 | 3 | 39.5 |
| exterior | 7 | 5 | 2 | 7 | 48.8 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 8 | 4 | 0 | 9 | 37.6 |
| edge | 4 | 3 | 2 | 1 | 59.7 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 4 | 1 | 1 | 6 | 61.8 |
| medium | 1 | 0 | 0 | 0 | 38.8 |
| large | 7 | 6 | 1 | 4 | 36.2 |

## Messages per drawing

1. high_quality_architectural-3691 (room (whole)): window symbol missing in the drawing -> added; wall added next to bedroom; wall added next to bedroom; wall added next to kitchen; missing room -> room drawn as other_room; wall added next to other_room; missing room -> room drawn as bedroom
2. colorful-14935 (room (whole)): sink added in bathroom
3. high_quality_architectural-8154 (room (whole)): nothing added
4. colorful-6226 (room (whole)): wall added next to bedroom; missing room -> room drawn as bedroom
5. high_quality_architectural-1376 (region): open doorway between bedroom and other_room had no door -> door added; wall added next to other_room; missing room -> room drawn as living
6. colorful-14938 (region): window added on the other_room wall; sink missing (detector) -> nothing added (the model kept the drawing as it was)
7. colorful-6628 (region): door symbol missing in the drawing -> added
8. high_quality-9975 (region): door symbol missing in the drawing -> added
9. high_quality_architectural-1654 (wall): sink added in bathroom
10. high_quality_architectural-1916 (room): window symbol missing in the drawing -> added
11. high_quality_architectural-1067 (window): door symbol missing in the drawing -> added; window symbol missing in the drawing -> added
12. high_quality_architectural-1359 (door (b)): window symbol missing in the drawing -> added

## Validity diagnosis: rules the fill added

Drawings with more violations after the fill than in the parse (omission region). Per rule: parse -> final (raw hybrid output before the type restriction): hybrid = the raw output has it too, restriction = only the restricted plan has it.

- 2. colorful-14935 (room (whole)): C16 0->1 (raw 0: restriction)

## Verdicts (one line per drawing; my reading of the figure and the numbers)

| # | omission | verdict |
|---|---|---|
| 1 | room (whole), large | **partly**: flagged (room + window flags at the merged room); walls redrawn on the wrong lines (IoU 0.15); a window marked on the outer wall there |
| 2 | room (whole), large | **wrong**: flagged only as a missing sink; a sink was added, no walls. Erasure caveat: on this grey-fill drawing the erased bathroom left a visible grey smudge |
| 3 | room (whole), medium | **missed**: nothing flagged |
| 4 | room (whole), large | **fixed**: room flag on the merged area; the dividing wall redrawn (IoU 0.46) |
| 5 | region, large (25%) | **wrong**: a door (R5) and one wall piece added inside a region that held a whole bathroom block; the parse already filled the region with one big room (tolerant mIoU 8) |
| 6 | region, large | **wrong**: window and sink flags around the region; a window added elsewhere (1 false addition), the region's walls not redrawn |
| 7 | region, large | **missed**: the only flag (door ink) is elsewhere; 1 false door mark |
| 8 | region, large | **wrong**: a door-ink flag on a door next to the region (counts as found, but nothing of the region is redrawn); 1 false door mark |
| 9 | wall (small) | **wrong**: flagged as a missing sink nearby; a sink added (1 false addition), the wall not restored |
| 10 | room walls (small) | **missed**: the only flag is a false window mark elsewhere |
| 11 | window (small) | **fixed, plus 1 false door mark**: window marked on the right spot (IoU 0.74) |
| 12 | door (b) (small) | **missed, plus 3 false window marks**: the window check fires on three outer walls of this drawing |

**Overall:** large omissions are found often (6 of 7 large ones have a flag on them) but rarely repaired: fixed 1 of 7
large, 0 of 1 medium. Flags on large omissions are mostly small-component types (sink, window, door) instead of
"room", so the fill adds one small symbol where a whole room or region is missing. The pipeline has no "region
missing" type: its fill works on <= 100 px windows with one type each, and the parser usually paints the erased area
as one big room, which the CHECK rules do not see as an error. 10 false additions, 6 of them window-ink marks.

## Size breakdown over all three batches (36 drawings; batch 1 = e2e_val_v4, batch 2, batch 3)

| size bin | n | found | fixed | fixed by addition | false adds | tol. mIoU before -> after | batches |
|---|---|---|---|---|---|---|---|
| small | 27 | 19 | 17 | 5 | 14 | 66.6 -> 67.5 | 1, 2, 3 |
| medium | 1 | 0 | 0 | 0 | 0 | 38.8 -> 38.8 | 3 |
| large | 8 | 7 | 2 | 1 | 4 | 37.6 -> 37.9 | 1, 3 |

Medium is nearly empty: room (whole) / region picks fell mostly in "large". The final test evaluation uses the
size-balanced generator (configs/stage2/test_eval.yaml), which targets each bin explicitly.

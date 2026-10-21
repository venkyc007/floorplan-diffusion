# End-to-end validation on 12 new held-out val drawings_batch3_v2

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall break at a door flag (src/stage2/detect/flag_gap.py).

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | room (whole) (room) | exterior | edge | large (9.072%) | window (WINK+detector); room (detector); room (INK+detector) | yes | yes | no (0.14) | no (0.06) | 0 | - | 54.6 -> 55.3 | no -> no | 115 |
| 2 | colorful-14935 | room (whole) (room) | interior | centre | large (7.86%) | sink (detector) | yes | no | no (0.03) | no (0.00) | 0 | - | 37.8 -> 37.4 | no -> no | 66 |
| 3 | high_quality_architectural-8154 | room (whole) (room) | interior | edge | medium (6.441%) | none | no | no | no (0.00) | no (0.00) | 0 | - | 38.8 -> 38.8 | no -> no | 4 |
| 4 | colorful-6226 | room (whole) (room) | exterior | edge | large (14.473%) | room (detector) | yes | yes | yes (0.46) | yes (0.32) | 0 | - | 47.8 -> 49.3 | no -> no | 71 |
| 5 | high_quality_architectural-1376 | region (room) | interior | centre | large (24.718%) | door (R5) | no | no | no (0.00) | no (0.00) | 1 | - | 8.2 -> 8.2 | no -> no | 66 |
| 6 | colorful-14938 | region (room) | exterior | centre | large (11.17%) | window (detector); sink (detector) | yes | no | no (0.02) | no (0.00) | 1 | - | 32.6 -> 32.6 | no -> no | 127 |
| 7 | colorful-6628 | region (room) | exterior | centre | large (13.917%) | door (INK) | no | no | no (0.00) | no (0.00) | 1 | - | 33.6 -> 33.6 | no -> no | 9 |
| 8 | high_quality-9975 | region (room) | exterior | centre | large (19.228%) | door (INK) | yes | no | no (0.01) | no (0.00) | 1 | - | 36.6 -> 36.6 | no -> no | 4 |
| 9 | high_quality_architectural-1654 | wall (wall) | interior | centre | small (0.313%) | sink (detector) | no | no | no (0.19) | no (0.00) | 1 | - | 50.5 -> 50.5 | no -> no | 59 |
| 10 | high_quality_architectural-1916 | room (room) | interior | centre | small (0.302%) | window (WINK) | no | no | no (0.00) | no (0.00) | 1 | - | 62.3 -> 62.3 | no -> no | 6 |
| 11 | high_quality_architectural-1067 | window (window) | exterior | edge | small (0.82%) | door (INK); window (detector) | yes | yes | yes (0.74) | no (0.00) | 1 | - | 95.3 -> 95.3 | no -> no | 6 |
| 12 | high_quality_architectural-1359 | door (b) (door) | exterior | centre | small (0.297%) | window (WINK); window (WINK); window (WINK) | no | no | no (0.54) | no (0.00) | 3 | - | 38.9 -> 38.9 | no -> no | 5 |

**Total:** found 6 / 12, right type 3 / 12, fixed 2 / 12 (by addition 1 / 12), false additions 10 (on 8 drawings); tolerant mIoU 44.8 -> 44.9; validity 0% -> 0%; time 45 s per drawing (max 127 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (b) | 1 | 0 | 0 | 3 | 38.9 |
| window | 1 | 1 | 1 | 1 | 95.3 |
| wall | 1 | 0 | 0 | 1 | 50.5 |
| room | 1 | 0 | 0 | 1 | 62.3 |
| room (whole) | 4 | 3 | 1 | 0 | 45.2 |
| region | 4 | 2 | 0 | 4 | 27.8 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 5 | 1 | 0 | 3 | 39.4 |
| exterior | 7 | 5 | 2 | 7 | 48.8 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 8 | 3 | 0 | 9 | 37.5 |
| edge | 4 | 3 | 2 | 1 | 59.7 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 4 | 1 | 1 | 6 | 61.8 |
| medium | 1 | 0 | 0 | 0 | 38.8 |
| large | 7 | 5 | 1 | 4 | 36.1 |

## Messages per drawing

1. high_quality_architectural-3691 (room (whole)): window symbol missing in the drawing -> added; wall added next to bedroom; wall added next to kitchen; missing room -> room drawn as other_room; wall added next to other_room; missing room -> room drawn as bedroom
2. colorful-14935 (room (whole)): sink added in bathroom
3. high_quality_architectural-8154 (room (whole)): nothing added
4. colorful-6226 (room (whole)): wall added next to bedroom; missing room -> room drawn as bedroom
5. high_quality_architectural-1376 (region): open doorway between bedroom and other_room had no door -> door added; wall added next to other_room
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

## Old vs new (same 12 drawings): e2e_val_batch3.csv -> e2e_val_batch3_v2.csv

| # | drawing | omission | old flags | new flags | fixed old -> new | fixed by addition old -> new | false adds old -> new | tol. mIoU after old -> new | valid after old -> new |
|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | room (whole) | window (WINK+detector); room (detector); room (INK+detector) | window (WINK+detector); room (detector); room (INK+detector) | no -> no | no -> no | 0 -> 0 | 55.3 -> 55.3 | no -> no |
| 2 | colorful-14935 | room (whole) | sink (detector) | sink (detector) | no -> no | no -> no | 0 -> 0 | 37.4 -> 37.4 | no -> no |
| 3 | high_quality_architectural-8154 | room (whole) | none | none | no -> no | no -> no | 0 -> 0 | 38.8 -> 38.8 | no -> no |
| 4 | colorful-6226 | room (whole) | room (detector) | room (detector) | yes -> yes | yes -> yes | 0 -> 0 | 49.3 -> 49.3 | no -> no |
| 5 | high_quality_architectural-1376 | region | door (R5); room (detector) | door (R5) | no -> no | no -> no | 1 -> 1 | 8.6 -> 8.2 | no -> no |
| 6 | colorful-14938 | region | window (detector); sink (detector) | window (detector); sink (detector) | no -> no | no -> no | 1 -> 1 | 32.6 -> 32.6 | no -> no |
| 7 | colorful-6628 | region | door (INK) | door (INK) | no -> no | no -> no | 1 -> 1 | 33.6 -> 33.6 | no -> no |
| 8 | high_quality-9975 | region | door (INK) | door (INK) | no -> no | no -> no | 1 -> 1 | 36.8 -> 36.6 | no -> no |
| 9 | high_quality_architectural-1654 | wall | sink (detector) | sink (detector) | no -> no | no -> no | 1 -> 1 | 50.5 -> 50.5 | no -> no |
| 10 | high_quality_architectural-1916 | room | window (WINK) | window (WINK) | no -> no | no -> no | 1 -> 1 | 62.3 -> 62.3 | no -> no |
| 11 | high_quality_architectural-1067 | window | door (INK); window (detector) | door (INK); window (detector) | yes -> yes | no -> no | 1 -> 1 | 95.3 -> 95.3 | no -> no |
| 12 | high_quality_architectural-1359 | door (b) | window (WINK); window (WINK); window (WINK) | window (WINK); window (WINK); window (WINK) | no -> no | no -> no | 3 -> 3 | 38.9 -> 38.9 | no -> no |

| totals | found | right type | fixed (new definition) | fixed by addition | false adds | tol. mIoU after | validity after | s per drawing |
|---|---|---|---|---|---|---|---|---|
| old | 7 | 4 | 2 | 1 | 10 | 44.9 | 0% | 15 |
| new | 6 | 3 | 2 | 1 | 10 | 44.9 | 0% | 45 |

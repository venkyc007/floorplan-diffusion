# End-to-end validation on 12 new held-out val drawings_batch2

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall break at a door flag (src/stage2/detect/flag_gap.py).

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-8299 | door (a) (door) | interior | centre | small (0.392%) | door (detector) | yes | yes | yes (0.58) | no (0.00) | 0 | - | 95.1 -> 95.1 | no -> no | 7 |
| 2 | high_quality-6141 | door (a) (door) | interior | centre | small (0.08%) | door (detector) | yes | yes | yes (0.71) | no (0.00) | 0 | - | 97.8 -> 97.8 | no -> no | 5 |
| 3 | high_quality_architectural-6514 | door (b) (door) | interior | centre | small (0.172%) | door (INK+detector) | yes | yes | yes (0.47) | no (0.00) | 0 | - | 58.4 -> 58.4 | no -> no | 2 |
| 4 | high_quality-4548 | door (b) (door) | interior | centre | small (0.095%) | none | no | no | no (0.72) | no (0.00) | 0 | - | 89.1 -> 89.1 | yes -> yes | 5 |
| 5 | high_quality_architectural-9491 | window (window) | exterior | edge | small (0.172%) | window (detector); wall (detector) | yes | yes | yes (0.61) | no (0.00) | 1 | - | 86.0 -> 86.0 | no -> no | 28 |
| 6 | high_quality_architectural-6301 | window (window) | exterior | edge | small (0.61%) | window (detector) | yes | yes | yes (0.40) | no (0.00) | 0 | - | 57.9 -> 57.9 | yes -> yes | 1 |
| 7 | high_quality_architectural-8532 | wall (wall) | interior | centre | small (0.134%) | none | no | no | no (0.11) | no (0.00) | 0 | - | 67.0 -> 67.0 | no -> no | 4 |
| 8 | high_quality-4708 | wall (wall) | interior | centre | small (0.053%) | window (WINK); wall (detector) | yes | yes | yes (0.73) | yes (0.91) | 1 | - | 51.8 -> 71.6 | no -> no | 44 |
| 9 | high_quality_architectural-865 | fixture (sink) | interior | edge | small (0.456%) | sink (detector) | no | no | no (0.00) | no (0.00) | 1 | - | 77.9 -> 77.9 | yes -> yes | 24 |
| 10 | high_quality_architectural-3794 | fixture (bath_shower) | interior | centre | small (0.043%) | none | no | no | no (0.46) | no (0.00) | 0 | - | 83.4 -> 83.4 | no -> no | 5 |
| 11 | high_quality_architectural-3096 | room (room) | interior | centre | small (0.321%) | room (detector); appliance (detector) | yes | yes | yes (0.31) | no (0.17) | 2 | - | 72.3 -> 37.3 | no -> no | 53 |
| 12 | high_quality_architectural-5566 | room (room) | interior | centre | small (1.042%) | room (detector); room (detector) | yes | yes | yes (0.52) | yes (0.35) | 0 | - | 81.7 -> 72.1 | no -> no | 49 |

**Total:** found 8 / 12, right type 8 / 12, fixed 8 / 12 (by addition 2 / 12), false additions 5 (on 4 drawings); tolerant mIoU 76.5 -> 74.5; validity 25% -> 25%; time 19 s per drawing (max 53 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (a) | 2 | 2 | 2 | 0 | 96.4 |
| door (b) | 2 | 1 | 1 | 0 | 73.8 |
| window | 2 | 2 | 2 | 1 | 72.0 |
| wall | 2 | 1 | 1 | 1 | 69.3 |
| fixture | 2 | 0 | 0 | 1 | 80.7 |
| room | 2 | 2 | 2 | 2 | 54.7 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 10 | 6 | 6 | 4 | 75.0 |
| exterior | 2 | 2 | 2 | 1 | 72.0 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 9 | 6 | 6 | 3 | 74.6 |
| edge | 3 | 2 | 2 | 2 | 73.9 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 12 | 8 | 8 | 5 | 74.5 |

## Messages per drawing

1. high_quality_architectural-8299 (door (a)): door symbol missing in the drawing -> added
2. high_quality-6141 (door (a)): door symbol missing in the drawing -> added
3. high_quality_architectural-6514 (door (b)): door symbol missing in the drawing -> added
4. high_quality-4548 (door (b)): nothing added
5. high_quality_architectural-9491 (window): window symbol missing in the drawing -> added; wall added next to living
6. high_quality_architectural-6301 (window): window symbol missing in the drawing -> added
7. high_quality_architectural-8532 (wall): nothing added
8. high_quality-4708 (wall): window symbol missing in the drawing -> added; wall added next to other_room
9. high_quality_architectural-865 (fixture): sink added in bathroom
10. high_quality_architectural-3794 (fixture): nothing added
11. high_quality_architectural-3096 (room): door added on the bathroom / hallway wall; wall added next to hallway; wall added next to hallway; wall added next to bathroom; wall added next to bathroom; missing room -> room drawn as bathroom; appliance added in kitchen
12. high_quality_architectural-5566 (room): wall added next to other_room; wall added next to hallway; wall added next to hallway; wall added next to hallway; missing room -> room drawn as bedroom; wall added next to bathroom; missing room -> room drawn as bathroom

## Validity diagnosis: rules the fill added

Drawings with more violations after the fill than in the parse (omission region). Per rule: parse -> final (raw hybrid output before the type restriction): hybrid = the raw output has it too, restriction = only the restricted plan has it.

- none

## Verdicts (one line per drawing; my reading of the figure and the numbers)

| # | omission | verdict |
|---|---|---|
| 1 | door (a) | **fixed**: the parser drew the door back from context; the detector flagged it -> "door symbol missing", marked in red on the right doorway |
| 2 | door (a) | **fixed**: as 1 |
| 3 | door (b) | **fixed**: ink + detector flag on the parsed door -> marked (IoU 0.47) |
| 4 | door (b) | **missed**: the parser redrew the door (IoU 0.72) but nothing flagged it; the drawing still lacks the leaf / arc |
| 5 | window | **fixed, plus 1 false wall**: window marked on the right spot; a detector "wall" flag elsewhere added a wall |
| 6 | window | **fixed**: window marked (IoU 0.40) |
| 7 | wall | **missed**: nothing flagged; the parse left the gap open |
| 8 | wall | **fixed, plus 1 false window**: wall redrawn by the fill (IoU 0.91 by addition); the window ink check marked a window elsewhere |
| 9 | fixture | **wrong**: a sink flag next to the erased fixture; a sink was added in the bathroom but not on the omission (1 false addition) |
| 10 | fixture | **missed**: nothing flagged |
| 11 | room | **partly, and worse overall**: dividing walls + a door redrawn (IoU 0.31, just over the bar) but the band is messy (tolerant mIoU 72 -> 37) and an appliance was added elsewhere (2 false additions) |
| 12 | room | **fixed, imperfect**: the hallway walls redrawn (IoU 0.52) but the region lost accuracy (81.7 -> 72.1) |

**Overall:** fixed 8 / 12 (only 2 of them by an addition: the rest are symbols the parser drew back and the CHECK
marked), 4 missed or wrong (door (b) 4, wall 7, fixtures 9 and 10). 5 false additions on 4 drawings. The fill
lowers tolerant mIoU in the omission region on the two room drawings.

## Batch 1 vs batch 2 (same pipeline: final CHECK with the window ink check, fill as of 2026-10-20 evening)

| batch | found | right type | fixed | fixed by addition | false adds | tol. mIoU before -> after | validity before -> after | s per drawing |
|---|---|---|---|---|---|---|---|---|
| 1 (e2e_val_v4, drawings seen while fixing) | 11 / 12 | 9 / 12 | 9 / 12 | 3 / 12 | 3 | 56.8 -> 60.9 | 33% -> 17% | 14 |
| 2 (fresh) | 8 / 12 | 8 / 12 | 8 / 12 | 2 / 12 | 5 | 76.5 -> 74.5 | 25% -> 25% | 19 |

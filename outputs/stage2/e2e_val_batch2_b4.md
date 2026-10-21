# End-to-end validation on 12 new held-out val drawings_batch2_b4

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall break at a door flag (src/stage2/detect/flag_gap.py).

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-8299 | door (a) (door) | interior | centre | small (0.392%) | door (detector) | yes | yes | yes (0.58) | no (0.00) | 0 | - | 95.1 -> 95.1 | no -> no | 8 |
| 2 | high_quality-6141 | door (a) (door) | interior | centre | small (0.08%) | door (detector) | yes | yes | yes (0.71) | no (0.00) | 0 | - | 97.8 -> 97.8 | no -> no | 7 |
| 3 | high_quality_architectural-6514 | door (b) (door) | interior | centre | small (0.172%) | door (INK+detector) | yes | yes | yes (0.47) | no (0.00) | 0 | - | 58.4 -> 58.4 | no -> no | 1 |
| 4 | high_quality-4548 | door (b) (door) | interior | centre | small (0.095%) | none | no | no | no (0.72) | no (0.00) | 0 | - | 89.1 -> 89.1 | yes -> yes | 5 |
| 5 | high_quality_architectural-9491 | window (window) | exterior | edge | small (0.172%) | window (detector); wall (detector) | yes | yes | yes (0.61) | no (0.00) | 1 | - | 86.0 -> 86.0 | no -> no | 24 |
| 6 | high_quality_architectural-6301 | window (window) | exterior | edge | small (0.61%) | window (detector) | yes | yes | yes (0.40) | no (0.00) | 0 | - | 57.9 -> 57.9 | yes -> yes | 1 |
| 7 | high_quality_architectural-8532 | wall (wall) | interior | centre | small (0.134%) | none | no | no | no (0.11) | no (0.00) | 0 | - | 67.0 -> 67.0 | no -> no | 3 |
| 8 | high_quality-4708 | wall (wall) | interior | centre | small (0.053%) | window (WINK); wall (detector) | yes | yes | yes (0.73) | yes (0.91) | 1 | - | 51.8 -> 71.6 | no -> no | 36 |
| 9 | high_quality_architectural-865 | fixture (sink) | interior | edge | small (0.456%) | sink (detector); sink (detector) | yes | yes | no (0.16) | no (0.16) | 1 | - | 77.9 -> 76.7 | yes -> no | 39 |
| 10 | high_quality_architectural-3794 | fixture (bath_shower) | interior | centre | small (0.043%) | none | no | no | no (0.46) | no (0.00) | 0 | - | 83.4 -> 83.4 | no -> no | 4 |
| 11 | high_quality_architectural-3096 | room (room) | interior | centre | small (0.321%) | room (detector); appliance (detector) | yes | yes | yes (0.31) | no (0.17) | 2 | - | 72.3 -> 37.3 | no -> no | 44 |
| 12 | high_quality_architectural-5566 | room (room) | interior | centre | small (1.042%) | room (detector); room (detector) | yes | yes | yes (0.52) | yes (0.35) | 0 | - | 81.7 -> 72.1 | no -> no | 40 |

**Total:** found 9 / 12, right type 9 / 12, fixed 8 / 12 (by addition 2 / 12), false additions 5 (on 4 drawings); tolerant mIoU 76.5 -> 74.4; validity 25% -> 17%; time 18 s per drawing (max 44 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (a) | 2 | 2 | 2 | 0 | 96.4 |
| door (b) | 2 | 1 | 1 | 0 | 73.8 |
| window | 2 | 2 | 2 | 1 | 72.0 |
| wall | 2 | 1 | 1 | 1 | 69.3 |
| fixture | 2 | 1 | 0 | 1 | 80.1 |
| room | 2 | 2 | 2 | 2 | 54.7 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 10 | 7 | 6 | 4 | 74.8 |
| exterior | 2 | 2 | 2 | 1 | 72.0 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 9 | 6 | 6 | 3 | 74.6 |
| edge | 3 | 3 | 2 | 2 | 73.5 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 12 | 9 | 8 | 5 | 74.4 |

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

- 9. high_quality_architectural-865 (fixture): C11 0->1 (raw 1: hybrid); C7 0->2 (raw 0: restriction); C8 0->1 (raw 0: restriction)

## Old vs new (same 12 drawings): e2e_val_batch2.csv -> e2e_val_batch2_b4.csv

| # | drawing | omission | old flags | new flags | fixed old -> new | fixed by addition old -> new | false adds old -> new | tol. mIoU after old -> new | valid after old -> new |
|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-8299 | door (a) | door (detector) | door (detector) | yes -> yes | no -> no | 0 -> 0 | 95.1 -> 95.1 | no -> no |
| 2 | high_quality-6141 | door (a) | door (detector) | door (detector) | yes -> yes | no -> no | 0 -> 0 | 97.8 -> 97.8 | no -> no |
| 3 | high_quality_architectural-6514 | door (b) | door (INK+detector) | door (INK+detector) | yes -> yes | no -> no | 0 -> 0 | 58.4 -> 58.4 | no -> no |
| 4 | high_quality-4548 | door (b) | none | none | no -> no | no -> no | 0 -> 0 | 89.1 -> 89.1 | yes -> yes |
| 5 | high_quality_architectural-9491 | window | window (detector); wall (detector) | window (detector); wall (detector) | yes -> yes | no -> no | 1 -> 1 | 86.0 -> 86.0 | no -> no |
| 6 | high_quality_architectural-6301 | window | window (detector) | window (detector) | yes -> yes | no -> no | 0 -> 0 | 57.9 -> 57.9 | yes -> yes |
| 7 | high_quality_architectural-8532 | wall | none | none | no -> no | no -> no | 0 -> 0 | 67.0 -> 67.0 | no -> no |
| 8 | high_quality-4708 | wall | window (WINK); wall (detector) | window (WINK); wall (detector) | yes -> yes | yes -> yes | 1 -> 1 | 71.6 -> 71.6 | no -> no |
| 9 | high_quality_architectural-865 | fixture | sink (detector) | sink (detector); sink (detector) | no -> no | no -> no | 1 -> 1 | 77.9 -> 76.7 | yes -> no |
| 10 | high_quality_architectural-3794 | fixture | none | none | no -> no | no -> no | 0 -> 0 | 83.4 -> 83.4 | no -> no |
| 11 | high_quality_architectural-3096 | room | room (detector); appliance (detector) | room (detector); appliance (detector) | yes -> yes | no -> no | 2 -> 2 | 37.3 -> 37.3 | no -> no |
| 12 | high_quality_architectural-5566 | room | room (detector); room (detector) | room (detector); room (detector) | yes -> yes | yes -> yes | 0 -> 0 | 72.1 -> 72.1 | no -> no |

| totals | found | right type | fixed (new definition) | fixed by addition | false adds | tol. mIoU after | validity after | s per drawing |
|---|---|---|---|---|---|---|---|---|
| old | 8 | 8 | 8 | 2 | 5 | 74.5 | 25% | 19 |
| new | 9 | 9 | 8 | 2 | 5 | 74.4 | 17% | 18 |

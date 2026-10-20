# End-to-end validation on 12 new held-out val drawings_v3

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

fixed = a flag there AND the final plan has the right class at the omission (IoU >= 0.3), whoever drew it; fixed by addition = the earlier definition (the fill's own additions only). gap = the measured wall break at a door flag (src/stage2/detect/flag_gap.py).

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | fixed by addition (IoU) | false adds | gap | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3546 | door (a) (door) | exterior | centre | small (0.217%) | door (INK); door (detector) | yes | yes | yes (0.58) | yes (0.58) | 1 | - | 23.2 -> 51.7 | yes -> no | 33 |
| 2 | high_quality_architectural-5625 | door (a) (door) | interior | centre | small (0.081%) | door (INK) | yes | yes | no (0.25) | no (0.00) | 0 | - | 65.2 -> 65.2 | yes -> yes | 4 |
| 3 | high_quality_architectural-3872 | door (b) (door) | interior | centre | small (0.159%) | door (INK) | yes | yes | yes (1.00) | no (0.00) | 0 | - | 99.6 -> 99.6 | no -> no | 1 |
| 4 | high_quality_architectural-1328 | door (b) (door) | interior | centre | small (0.045%) | door (detector) | yes | yes | yes (0.70) | no (0.00) | 0 | - | 97.2 -> 97.2 | no -> no | 11 |
| 5 | high_quality_architectural-4900 | window (window) | exterior | edge | small (0.43%) | window (detector) | yes | yes | yes (0.95) | no (0.00) | 0 | - | 99.8 -> 99.8 | no -> no | 2 |
| 6 | high_quality_architectural-6347 | window (window) | exterior | edge | large (7.882%) | window (detector) | yes | yes | yes (0.72) | no (0.00) | 0 | - | 49.4 -> 49.4 | no -> no | 1 |
| 7 | high_quality-13852 | wall (wall) | exterior | edge | small (0.194%) | room (detector) | yes | no | no (0.17) | no (0.23) | 0 | - | 55.5 -> 52.7 | yes -> no | 25 |
| 8 | high_quality_architectural-6339 | wall (wall) | interior | centre | small (0.283%) | door (INK) | yes | no | yes (0.31) | no (0.00) | 0 | - | 45.2 -> 45.2 | no -> no | 2 |
| 9 | high_quality_architectural-9237 | fixture (appliance) | interior | edge | small (0.588%) | none | no | no | no (0.00) | no (0.00) | 0 | - | 53.0 -> 53.0 | no -> no | 1 |
| 10 | high_quality_architectural-1186 | fixture (appliance) | interior | centre | small (0.191%) | appliance (detector) | yes | yes | yes (0.53) | yes (0.53) | 0 | - | 17.2 -> 37.4 | yes -> no | 27 |
| 11 | high_quality_architectural-5559 | room (room) | exterior | centre | small (0.876%) | door (C8); room (detector) | yes | yes | yes (0.55) | yes (0.47) | 1 | - | 27.5 -> 29.4 | no -> no | 49 |
| 12 | colorful-3134 | room (room) | interior | centre | small (0.78%) | room (detector) | yes | yes | yes (0.31) | no (0.29) | 0 | - | 48.7 -> 47.9 | no -> no | 24 |

**Total:** found 11 / 12, right type 9 / 12, fixed 9 / 12 (by addition 3 / 12), false additions 2 (on 2 drawings); tolerant mIoU 56.8 -> 60.7; validity 33% -> 8%; time 15 s per drawing (max 49 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (a) | 2 | 2 | 1 | 1 | 58.5 |
| door (b) | 2 | 2 | 2 | 0 | 98.4 |
| window | 2 | 2 | 2 | 0 | 74.6 |
| wall | 2 | 2 | 1 | 0 | 49.0 |
| fixture | 2 | 1 | 1 | 0 | 45.2 |
| room | 2 | 2 | 2 | 1 | 38.6 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 7 | 6 | 5 | 0 | 63.6 |
| exterior | 5 | 5 | 4 | 2 | 56.6 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 8 | 8 | 7 | 2 | 59.2 |
| edge | 4 | 3 | 2 | 0 | 63.7 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 11 | 10 | 8 | 2 | 61.7 |
| large | 1 | 1 | 1 | 0 | 49.4 |

## Messages per drawing

1. high_quality_architectural-3546 (door (a)): door symbol missing in the drawing -> added; door added on the other_room / background wall; wall added next to the plan
2. high_quality_architectural-5625 (door (a)): door symbol missing in the drawing -> added
3. high_quality_architectural-3872 (door (b)): door symbol missing in the drawing -> added
4. high_quality_architectural-1328 (door (b)): door symbol missing in the drawing -> added
5. high_quality_architectural-4900 (window): window symbol missing in the drawing -> added
6. high_quality_architectural-6347 (window): window symbol missing in the drawing -> added
7. high_quality-13852 (wall): door added on a wall; wall added next to kitchen; wall added next to kitchen; missing room -> room drawn as kitchen
8. high_quality_architectural-6339 (wall): door symbol missing in the drawing -> added
9. high_quality_architectural-9237 (fixture): nothing added
10. high_quality_architectural-1186 (fixture): appliance added in kitchen
11. high_quality_architectural-5559 (room): door missing (C8) -> nothing added (the model kept the drawing as it was); door added on the hallway / outdoor wall; wall added next to the plan; wall added next to hallway; wall added next to bathroom; missing room -> room drawn as kitchen
12. colorful-3134 (room): door added on the bathroom / hallway wall; wall added next to kitchen; wall added next to bathroom; missing room -> room drawn as bathroom

## Validity diagnosis: rules the fill added

Drawings with more violations after the fill than in the parse (omission region). Per rule: parse -> final (raw hybrid output before the type restriction): hybrid = the raw output has it too, restriction = only the restricted plan has it.

- 1. high_quality_architectural-3546 (door (a)): C9 0->1 (raw 1: hybrid)
- 7. high_quality-13852 (wall): C13 0->1 (raw 1: hybrid); C14 0->1 (raw 1: hybrid); C7 0->1 (raw 1: hybrid)
- 10. high_quality_architectural-1186 (fixture): C16 0->1 (raw 1: hybrid); C7 1->2 (raw 2: hybrid)

## Old vs new (same 12 drawings)

Old = before the 2026-10-20 fixes (eta 0.1; its 'fixed' counted only the fill's additions). New = eta 0, symbol-already-in-parse, door at the flag point, wall-gap rule, one room type, door + adjacent walls.

| # | drawing | omission | old flags | new flags | old fixed by addition | new fixed by addition | new fixed | old -> new false adds | old -> new tol. mIoU after | old -> new valid after |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3546 | door (a) | door (INK); door (detector) | door (INK); door (detector) | yes | yes | yes | 1 -> 1 | 49.6 -> 51.7 | no -> no |
| 2 | high_quality_architectural-5625 | door (a) | door (INK) | door (INK) | no | no | no | 0 -> 0 | 63.9 -> 65.2 | yes -> yes |
| 3 | high_quality_architectural-3872 | door (b) | door (INK) | door (INK) | no | no | yes | 0 -> 0 | 99.6 -> 99.6 | no -> no |
| 4 | high_quality_architectural-1328 | door (b) | door (detector) | door (detector) | no | no | yes | 0 -> 0 | 97.2 -> 97.2 | no -> no |
| 5 | high_quality_architectural-4900 | window | window (detector) | window (detector) | no | no | yes | 0 -> 0 | 99.8 -> 99.8 | no -> no |
| 6 | high_quality_architectural-6347 | window | window (detector) | window (detector) | no | no | yes | 0 -> 0 | 55.4 -> 49.4 | no -> no |
| 7 | high_quality-13852 | wall | room (detector) | room (detector) | no | no | no | 0 -> 0 | 52.7 -> 52.7 | no -> no |
| 8 | high_quality_architectural-6339 | wall | door (INK) | door (INK) | no | no | yes | 0 -> 0 | 31.5 -> 45.2 | no -> no |
| 9 | high_quality_architectural-9237 | fixture | none | none | no | no | no | 0 -> 0 | 53.0 -> 53.0 | no -> no |
| 10 | high_quality_architectural-1186 | fixture | appliance (detector) | appliance (detector) | yes | yes | yes | 0 -> 0 | 37.9 -> 37.4 | no -> no |
| 11 | high_quality_architectural-5559 | room | door (C8); room (detector) | door (C8); room (detector) | yes | yes | yes | 1 -> 1 | 29.4 -> 29.4 | no -> no |
| 12 | colorful-3134 | room | room (detector) | room (detector) | no | no | yes | 0 -> 0 | 40.8 -> 47.9 | no -> no |

| totals | found | right type | fixed (new definition) | fixed by addition | false adds | tol. mIoU after | validity after | s per drawing |
|---|---|---|---|---|---|---|---|---|
| old | 11 | 9 | - | 3 | 2 | 59.2 | 8% | 27 |
| new | 11 | 9 | 9 | 3 | 2 | 60.7 | 8% | 15 |

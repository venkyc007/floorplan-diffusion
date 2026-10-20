# End-to-end validation on 12 new held-out val drawings

Pipeline: parse v2B -> final CHECK -> fill (fast hybrid, t0 = 300, N = 2, eta = 0.1, sketch prior, type restriction); Kaggle CPU. One omission per drawing (erasure v2). Drawings: held-out val plans minus the check set, fixed random order (seed 20). Scores at the parsed map's scale; tolerant mIoU / validity inside the omission region (box + 4 px), before -> after the fill, vs the ground truth.

| # | drawing | omission | location | position | size | flags (type + source) | found | right type | fixed (IoU) | false adds | tol. mIoU before -> after | valid before -> after | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3546 | door (a) (door) | exterior | centre | small (0.217%) | door (INK); door (detector) | yes | yes | yes (0.55) | 1 | 23.2 -> 49.6 | yes -> no | 53 |
| 2 | high_quality_architectural-5625 | door (a) (door) | interior | centre | small (0.081%) | door (INK) | yes | yes | no (0.03) | 0 | 65.2 -> 63.9 | yes -> yes | 26 |
| 3 | high_quality_architectural-3872 | door (b) (door) | interior | centre | small (0.159%) | door (INK) | yes | yes | no (0.00) | 0 | 99.6 -> 99.6 | no -> no | 23 |
| 4 | high_quality_architectural-1328 | door (b) (door) | interior | centre | small (0.045%) | door (detector) | yes | yes | no (0.00) | 0 | 97.2 -> 97.2 | no -> no | 31 |
| 5 | high_quality_architectural-4900 | window (window) | exterior | edge | small (0.43%) | window (detector) | yes | yes | no (0.00) | 0 | 99.8 -> 99.8 | no -> no | 24 |
| 6 | high_quality_architectural-6347 | window (window) | exterior | edge | large (7.882%) | window (detector) | yes | yes | no (0.16) | 0 | 49.4 -> 55.4 | no -> no | 22 |
| 7 | high_quality-13852 | wall (wall) | exterior | edge | small (0.194%) | room (detector) | yes | no | no (0.23) | 0 | 55.5 -> 52.7 | yes -> no | 24 |
| 8 | high_quality_architectural-6339 | wall (wall) | interior | centre | small (0.283%) | door (INK) | yes | no | no (0.00) | 0 | 45.2 -> 31.5 | no -> no | 24 |
| 9 | high_quality_architectural-9237 | fixture (appliance) | interior | edge | small (0.588%) | none | no | no | no (0.00) | 0 | 53.0 -> 53.0 | no -> no | 1 |
| 10 | high_quality_architectural-1186 | fixture (appliance) | interior | centre | small (0.191%) | appliance (detector) | yes | yes | yes (0.54) | 0 | 17.2 -> 37.9 | yes -> no | 26 |
| 11 | high_quality_architectural-5559 | room (room) | exterior | centre | small (0.876%) | door (C8); room (detector) | yes | yes | yes (0.47) | 1 | 27.5 -> 29.4 | no -> no | 46 |
| 12 | colorful-3134 | room (room) | interior | centre | small (0.78%) | room (detector) | yes | yes | no (0.29) | 0 | 48.7 -> 40.8 | no -> no | 23 |

**Total:** found 11 / 12, right type 9 / 12, fixed 3 / 12, false additions 2 (on 2 drawings); tolerant mIoU 56.8 -> 59.2; validity 33% -> 8%; time 27 s per drawing (max 53 s).

## Breakdown by omission metadata

By type:

| type | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| door (a) | 2 | 2 | 1 | 1 | 56.8 |
| door (b) | 2 | 2 | 0 | 0 | 98.4 |
| window | 2 | 2 | 0 | 0 | 77.6 |
| wall | 2 | 2 | 0 | 0 | 42.1 |
| fixture | 2 | 1 | 1 | 0 | 45.5 |
| room | 2 | 2 | 1 | 1 | 35.1 |

By location:

| location | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| interior | 7 | 6 | 1 | 0 | 60.6 |
| exterior | 5 | 5 | 2 | 2 | 57.4 |

By position:

| position | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| centre | 8 | 8 | 3 | 2 | 56.2 |
| edge | 4 | 3 | 0 | 0 | 65.2 |

By size_bin:

| size_bin | n | found | fixed | false adds | tol. mIoU after |
|---|---|---|---|---|---|
| small | 11 | 10 | 3 | 2 | 59.6 |
| large | 1 | 1 | 0 | 0 | 55.4 |

## Messages per drawing

1. high_quality_architectural-3546 (door (a)): door added on the bedroom / other_room wall; door added on the other_room / background wall
2. high_quality_architectural-5625 (door (a)): door added on the bathroom / other_room wall
3. high_quality_architectural-3872 (door (b)): door missing (INK) -> nothing added (the model kept the drawing as it was)
4. high_quality_architectural-1328 (door (b)): door missing (detector) -> nothing added (the model kept the drawing as it was)
5. high_quality_architectural-4900 (window): window missing (detector) -> nothing added (the model kept the drawing as it was)
6. high_quality_architectural-6347 (window): window added on the living wall
7. high_quality-13852 (wall): door added on the the wall; wall added next to kitchen; wall added next to kitchen; missing room -> room drawn as kitchen
8. high_quality_architectural-6339 (wall): door added on the bedroom / hallway wall
9. high_quality_architectural-9237 (fixture): nothing added
10. high_quality_architectural-1186 (fixture): appliance added in kitchen
11. high_quality_architectural-5559 (room): door missing (C8) -> nothing added (the model kept the drawing as it was); door added on the hallway / outdoor wall; wall added next to the plan; wall added next to hallway; wall added next to bathroom; missing room -> room drawn as kitchen
12. colorful-3134 (room): door added on the bathroom / hallway wall; wall added next to kitchen; wall added next to bathroom; missing room -> room drawn as living, bedroom, kitchen, bathroom, hallway

## Verdicts (one line per drawing; my reading of the figure and the numbers)

| # | omission | verdict |
|---|---|---|
| 1 | door (a) | **fixed, plus 1 false door** (ink flag elsewhere, on an outer wall) |
| 2 | door (a) | **wrong**: flagged in the right place, but the door was drawn on the neighbouring bathroom wall, not across the closed doorway (IoU 0.03) |
| 3 | door (b) | **plan right, nothing to add**: the parser redrew the erased door (mIoU 99.6 before the fill), so "not fixed" only means no addition was needed |
| 4 | door (b) | **plan right, nothing to add**: as 3 (parser redrew it, 97.2) |
| 5 | window | **plan right, nothing to add**: as 3 (parser redrew it, 99.8) |
| 6 | window | **partly**: window added on the right outer wall but shorter than the erased run (IoU 0.16; the omitted "window" is one long glazed wall, 7.9% of the building) |
| 7 | wall | **wrong type**: found, but flagged as a room; walls + a door redrawn around the gap, the wall itself only partly (IoU 0.23) |
| 8 | wall | **wrong**: the gap was read as a doorway (ink flag) and a door was put in it; tolerant mIoU 45 -> 32 |
| 9 | fixture | **missed**: nothing flagged |
| 10 | fixture | **fixed**: appliance redrawn in the kitchen (IoU 0.54), no false additions |
| 11 | room | **partly**: dividing walls redrawn (IoU 0.47), plus 1 false door on the hallway / outdoor wall |
| 12 | room | **partly**: some dividing walls back (IoU 0.29, just under the bar) and a door; room types in the band mixed |

**Overall:** found 11 / 12, right type 9 / 12. Fixed by an addition: 3 / 12. Counting the 3 drawings where the parser
already redrew the omission, the final plan is right at the omission in 6 / 12. 2 false additions in total.
Validity is low already before the fill (4 / 12): the parse itself has violations near the omission. The fill
adds violations in 5 drawings and removes them in 3. Time: 27 s per drawing on the Kaggle CPU (max 53 s).

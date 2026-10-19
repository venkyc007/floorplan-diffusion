# Parser v2 vs v1 (held-out val drawings)

159 held-out val drawings (configs/stage2/val_split/heldout_ids.txt; none used for any choice). Text erased = src/stage2/parse/text_erase.py with the ground-truth labels. v2A erases text itself before parsing (two passes); v1 and v2B parse once. Numbers in %.

| model | input | mIoU | IoU wall | IoU door | IoU window | IoU railing | F1 door | F1 window | F1 toilet | F1 sink | F1 bath_shower | F1 appliance | IoU living | IoU bedroom | IoU kitchen | IoU bathroom | IoU hallway | IoU other_room | **room-type mIoU** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | normal | 70.7 | 77.4 | 65.6 | 74.6 | 41.3 | 78.6 | 74.5 | 57.2 | 82.0 | 38.3 | 66.8 | 87.7 | 87.5 | 79.2 | 72.5 | 68.9 | 73.0 | **78.1** |
| v1 | text erased | 61.6 | 77.1 | 62.9 | 73.8 | 39.7 | 75.6 | 74.9 | 60.1 | 78.6 | 33.5 | 71.3 | 65.2 | 65.7 | 68.9 | 46.7 | 34.8 | 48.1 | **54.9** |
| v2A (no text) | normal | 65.3 | 77.0 | 65.4 | 75.6 | 41.4 | 77.2 | 77.3 | 55.7 | 82.1 | 36.6 | 65.9 | 71.3 | 76.6 | 72.1 | 63.2 | 56.4 | 50.6 | **65.0** |
| v2A (no text) | text erased | 66.9 | 77.9 | 65.7 | 75.9 | 41.2 | 78.1 | 77.7 | 58.7 | 83.6 | 36.2 | 74.3 | 72.8 | 75.7 | 74.1 | 64.3 | 56.9 | 49.5 | **65.5** |
| v2B (mixed) | normal | 70.1 | 77.6 | 66.6 | 75.8 | 42.6 | 79.5 | 77.3 | 56.2 | 83.1 | 39.5 | 66.0 | 87.4 | 85.9 | 79.1 | 68.4 | 67.2 | 69.4 | **76.3** |
| v2B (mixed) | text erased | 66.2 | 77.9 | 65.3 | 76.1 | 41.1 | 78.3 | 77.3 | 58.7 | 83.2 | 37.0 | 72.1 | 71.6 | 73.2 | 72.3 | 63.5 | 56.9 | 48.7 | **64.3** |

## Room-type confusion: v1, normal (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 94 | 0 | 2 | 0 | 1 | 2 | 1 |
| bedroom | 0 | 93 | 0 | 0 | 0 | 4 | 3 |
| kitchen | 3 | 0 | 89 | 0 | 2 | 2 | 5 |
| bathroom | 0 | 0 | 0 | 88 | 0 | 4 | 8 |
| hallway | 2 | 0 | 1 | 0 | 87 | 4 | 5 |
| other_room | 3 | 3 | 2 | 3 | 4 | 79 | 6 |

## Room-type confusion: v1, text erased (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 75 | 1 | 5 | 0 | 1 | 16 | 2 |
| bedroom | 2 | 75 | 1 | 0 | 0 | 20 | 3 |
| kitchen | 5 | 0 | 86 | 0 | 0 | 4 | 5 |
| bathroom | 0 | 0 | 0 | 57 | 0 | 33 | 10 |
| hallway | 1 | 0 | 2 | 0 | 41 | 49 | 6 |
| other_room | 7 | 8 | 3 | 3 | 2 | 71 | 7 |

## Room-type confusion: v2A (no text), normal (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 87 | 1 | 1 | 0 | 1 | 3 | 7 |
| bedroom | 3 | 87 | 0 | 0 | 0 | 4 | 6 |
| kitchen | 8 | 0 | 82 | 0 | 2 | 1 | 6 |
| bathroom | 0 | 0 | 0 | 85 | 0 | 5 | 10 |
| hallway | 2 | 0 | 1 | 0 | 82 | 8 | 8 |
| other_room | 10 | 8 | 3 | 5 | 9 | 56 | 10 |

## Room-type confusion: v2A (no text), text erased (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 91 | 1 | 2 | 0 | 1 | 4 | 1 |
| bedroom | 3 | 90 | 0 | 0 | 0 | 4 | 3 |
| kitchen | 9 | 0 | 84 | 0 | 2 | 1 | 4 |
| bathroom | 0 | 0 | 0 | 87 | 0 | 6 | 7 |
| hallway | 1 | 0 | 1 | 0 | 83 | 9 | 6 |
| other_room | 11 | 11 | 2 | 5 | 9 | 55 | 7 |

## Room-type confusion: v2B (mixed), normal (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 96 | 0 | 1 | 0 | 1 | 1 | 1 |
| bedroom | 0 | 94 | 0 | 0 | 0 | 3 | 3 |
| kitchen | 5 | 0 | 88 | 0 | 2 | 1 | 5 |
| bathroom | 0 | 0 | 0 | 89 | 0 | 3 | 8 |
| hallway | 3 | 0 | 1 | 0 | 87 | 4 | 6 |
| other_room | 4 | 5 | 2 | 4 | 5 | 74 | 6 |

## Room-type confusion: v2B (mixed), text erased (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 88 | 2 | 3 | 0 | 1 | 5 | 1 |
| bedroom | 3 | 90 | 0 | 0 | 0 | 5 | 3 |
| kitchen | 8 | 0 | 85 | 0 | 2 | 1 | 4 |
| bathroom | 0 | 0 | 0 | 86 | 0 | 6 | 8 |
| hallway | 2 | 0 | 1 | 0 | 79 | 12 | 6 |
| other_room | 11 | 13 | 2 | 5 | 7 | 55 | 6 |

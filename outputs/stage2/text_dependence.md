# Room types: does the parser read the text? (parser v1, val)

100 typed val drawings (file order). Text erased: dark character-sized pieces not touching walls, openings, railings or fixtures, inpainted (median 987 px per drawing). Examples: outputs/stage2/text_erasure_examples.png.

| room type IoU | living | bedroom | kitchen | bathroom | hallway | other_room | mean |
|---|---|---|---|---|---|---|---|
| original | 89.2 | 89.4 | 80.0 | 77.3 | 77.1 | 73.5 | 81.1 |
| text erased | 64.3 | 61.1 | 73.9 | 36.8 | 23.5 | 43.9 | 50.6 |

Control (did the erasure damage structure?): original: wall / door / window IoU 80.5 / 67.6 / 75.5; text erased: wall / door / window IoU 80.5 / 65.6 / 74.9.

**Room-type mIoU drop when text is erased: 30.5 points.**

## Confusion, original (% of each true type's pixels)

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 94 | 0 | 1 | 0 | 1 | 2 | 1 |
| bedroom | 0 | 93 | 0 | 0 | 0 | 6 | 1 |
| kitchen | 2 | 0 | 91 | 0 | 1 | 2 | 4 |
| bathroom | 0 | 0 | 0 | 90 | 0 | 5 | 5 |
| hallway | 1 | 0 | 1 | 0 | 92 | 1 | 5 |
| other_room | 3 | 2 | 2 | 2 | 3 | 81 | 7 |

## Confusion, text erased

| true \ predicted | living | bedroom | kitchen | bathroom | hallway | other_room | not a room |
|---|---|---|---|---|---|---|---|
| living | 71 | 0 | 4 | 0 | 1 | 20 | 3 |
| bedroom | 2 | 68 | 0 | 0 | 0 | 28 | 2 |
| kitchen | 1 | 0 | 90 | 0 | 0 | 3 | 5 |
| bathroom | 0 | 0 | 0 | 42 | 0 | 50 | 7 |
| hallway | 1 | 0 | 1 | 0 | 25 | 67 | 7 |
| other_room | 6 | 7 | 3 | 2 | 1 | 73 | 7 |

## Proposal (not launched): parser v2

The parser leans on room names. Proposal: parser v2 = fine-tune v1 with random text erasure as an augmentation (this module's text mask + inpainting, applied to ~50% of training drawings) plus the local constraint losses, as planned for v2 in CLAUDE.md. Same data and split; select on val room-type mIoU measured on text-erased drawings.

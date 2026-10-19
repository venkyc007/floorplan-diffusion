# Check set: 2 / 12 correct (fast mode, CPU)

found = a hole overlaps the omission; restored = the expected class was added there; extra = added components elsewhere; correct = restored and no extra (intact: nothing added).

| # | drawing | omission | found | restored | extra | correct | holes | s |
|---|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | yes | yes | 0 | **yes** | 1 | 28 |
| 2 | colorful-1148 | door (a) | yes | yes | 3 | no | 3 | 60 |
| 3 | colorful-1261 | door (b) | no | no | 1 | no | 1 | 24 |
| 4 | colorful-13689 | door (b) | no | no | 0 | no | 0 | 2 |
| 5 | colorful-13513 | window | no | no | 1 | no | 1 | 19 |
| 6 | colorful-13804 | window | no | no | 0 | no | 0 | 1 |
| 7 | colorful-13820 | wall | no | no | 6 | no | 7 | 132 |
| 8 | colorful-13824 | wall | no | no | 0 | no | 0 | 3 |
| 9 | colorful-13826 | fixture | no | no | 0 | no | 0 | 4 |
| 10 | colorful-14036 | fixture | no | no | 0 | no | 0 | 1 |
| 11 | colorful-14126 | room | no | no | 0 | no | 0 | 1 |
| 12 | colorful-14147 | intact | no | no | 0 | **yes** | 0 | 2 |

## 1. colorful-11261 (door (a)): automatic verdict correct

Removed: door (a). Detected: door (INK). Found yes, restored yes, extra 0.

Added (messages):

- door added on the bedroom / kitchen wall

## 2. colorful-1148 (door (a)): automatic verdict wrong

Removed: door (a). Detected: door (C8), door (INK), door (INK). Found yes, restored yes, extra 3.

Added (messages):

- bedroom had no door -> door added on the bedroom / hallway wall
- door added on the other_room / outdoor wall
- door added on the other_room / outdoor wall
- door added on the other_room / background wall

## 3. colorful-1261 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (R5). Found no, restored no, extra 1.

Added (messages):

- open doorway between hallway and other_room had no door -> door added

## 4. colorful-13689 (door (b)): automatic verdict wrong

Removed: door (b). Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 5. colorful-13513 (window): automatic verdict wrong

Removed: window. Detected: door (INK). Found no, restored no, extra 1.

Added (messages):

- door added on the hallway / outdoor wall

## 6. colorful-13804 (window): automatic verdict wrong

Removed: window. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 7. colorful-13820 (wall): automatic verdict wrong

Removed: wall. Detected: door (INK), door (INK), door (INK), door (INK), door (INK), door (INK), door (INK). Found no, restored no, extra 6.

Added (messages):

- door added on the bedroom / other_room wall
- door added on the the wall
- door added on the living / hallway wall
- door added on the living / outdoor wall
- door missing (INK) -> nothing added (the model kept the drawing as it was)
- door added on the other_room / other_room wall

## 8. colorful-13824 (wall): automatic verdict wrong

Removed: wall. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 9. colorful-13826 (fixture): automatic verdict wrong

Removed: fixture. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 10. colorful-14036 (fixture): automatic verdict wrong

Removed: fixture. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 11. colorful-14126 (room): automatic verdict wrong

Removed: room. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 12. colorful-14147 (intact): automatic verdict correct

Removed: intact. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## Holes in the right place: raw hybrid output (step 3)

Per hole overlapping the omission: new pixels of the hole's type in the RAW hybrid output (before the type restriction), its largest new component, what the restriction kept, and the classes the hybrid changed inside the box. Figure: check_set_holes.png.

| # | drawing | omission | hole | raw new px of type (largest) | kept px | hybrid changed (class: px) | why nothing / what |
|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | door (INK) | 11 (11) | 11 | bedroom: 12, door: 11 | door kept (11 px) on the omitted spot |
| 2 | colorful-1148 | door (a) | door (C8) | 105 (105) | 39 | door: 105 | door kept (39 px) on the omitted spot |

## My verdicts (detector v2, re-tuned CHECK, ink guard, door clipping; after looking at check_set.png and check_set_holes.png)

| # | omission | verdict | why |
|---|---|---|---|
| 1 | door (a) | correct | the ink check found the door the parser had drawn back without a symbol; a door was drawn at the erased doorway (on the straight wall next to the angled opening) |
| 2 | door (a) | partly | C8 found the cut-off bedroom; the door is now clipped to the right wall at median length (39 px, was 105 L-shaped); but false ink holes added 3 doors on exterior walls |
| 3 | door (b) | wrong | R5 fired at another gap and added a door there; the erased doorway was not found |
| 4 | door (b) | wrong | nothing flagged |
| 5 | window | wrong | not found (the detector is off); a false ink hole added a door on an exterior wall |
| 6 | window | wrong | nothing flagged |
| 7 | wall | wrong | 7 false ink holes (the guard did not trigger: fewer than half of this drawing's doors lack arcs), 6 doors added |
| 8 | wall | wrong | nothing flagged |
| 9 | fixture | wrong | not flagged (detector v2 at 0.98 / 160 px proposes nothing here) |
| 10 | fixture | wrong | nothing flagged |
| 11 | room | wrong | nothing flagged |
| 12 | intact | correct | nothing flagged, nothing added |

**Score: 2 / 12 correct, 1 partly.** Previous run (detector off): 2 / 12, 1 partly. Settings were not changed based on this set.

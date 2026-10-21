# Check set: 5 / 12 correct (fast mode, CPU)

found = a hole overlaps the omission; restored = the expected class was added there; extra = added components elsewhere; correct = restored and no extra (intact: nothing added).

| # | drawing | omission | found | restored | extra | correct | holes | s |
|---|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | yes | yes | 1 | no | 1 | 25 |
| 2 | colorful-1148 | door (a) | yes | yes | 0 | **yes** | 1 | 22 |
| 3 | colorful-1261 | door (b) | no | no | 1 | no | 1 | 33 |
| 4 | colorful-13689 | door (b) | yes | no | 0 | no | 1 | 20 |
| 5 | colorful-13513 | window | no | no | 0 | no | 0 | 1 |
| 6 | colorful-13804 | window | yes | yes | 0 | **yes** | 1 | 19 |
| 7 | colorful-13820 | wall | yes | yes | 3 | no | 5 | 93 |
| 8 | colorful-13824 | wall | yes | yes | 0 | **yes** | 1 | 24 |
| 9 | colorful-13826 | fixture | yes | yes | 1 | no | 2 | 42 |
| 10 | colorful-14036 | fixture | no | no | 0 | no | 0 | 1 |
| 11 | colorful-14126 | room | yes | yes | 0 | **yes** | 3 | 56 |
| 12 | colorful-14147 | intact | no | no | 0 | **yes** | 0 | 2 |

## 1. colorful-11261 (door (a)): automatic verdict wrong

Removed: door (a). Detected: room (INK+detector). Found yes, restored yes, extra 1.

Added (messages):

- door added on the bedroom / hallway wall
- wall added next to hallway
- wall added next to hallway
- wall added next to hallway
- missing room -> room drawn as bedroom, kitchen, bathroom

## 2. colorful-1148 (door (a)): automatic verdict correct

Removed: door (a). Detected: door (C8+detector). Found yes, restored yes, extra 0.

Added (messages):

- bedroom had no door -> door added on the bedroom / hallway wall

## 3. colorful-1261 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (R5). Found no, restored no, extra 1.

Added (messages):

- open doorway between hallway and other_room had no door -> door added

## 4. colorful-13689 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (detector). Found yes, restored no, extra 0.

Added (messages):

- door missing (detector) -> nothing added (the model kept the drawing as it was)

## 5. colorful-13513 (window): automatic verdict wrong

Removed: window. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 6. colorful-13804 (window): automatic verdict correct

Removed: window. Detected: window (detector). Found yes, restored yes, extra 0.

Added (messages):

- window added on the kitchen wall

## 7. colorful-13820 (wall): automatic verdict wrong

Removed: wall. Detected: door (detector), room (detector), door (INK), door (INK), door (INK). Found yes, restored yes, extra 3.

Added (messages):

- door added on the the wall
- wall added next to other_room
- wall added next to other_room
- door added on the bedroom / other_room wall
- door missing (INK) -> nothing added (the model kept the drawing as it was)

## 8. colorful-13824 (wall): automatic verdict correct

Removed: wall. Detected: wall (detector). Found yes, restored yes, extra 0.

Added (messages):

- wall added next to other_room

## 9. colorful-13826 (fixture): automatic verdict wrong

Removed: fixture. Detected: sink (detector), appliance (detector). Found yes, restored yes, extra 1.

Added (messages):

- sink added in bathroom
- appliance added in kitchen

## 10. colorful-14036 (fixture): automatic verdict wrong

Removed: fixture. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 11. colorful-14126 (room): automatic verdict correct

Removed: room. Detected: room (detector), room (detector), room (detector). Found yes, restored yes, extra 0.

Added (messages):

- wall added next to bedroom
- missing room -> room drawn as kitchen
- wall added next to kitchen
- wall added next to kitchen
- missing room -> room drawn as living, bedroom
- wall added next to living

## 12. colorful-14147 (intact): automatic verdict correct

Removed: intact. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## Holes in the right place: raw hybrid output (step 3)

Per hole overlapping the omission: new pixels of the hole's type in the RAW hybrid output (before the type restriction), its largest new component, what the restriction kept, and the classes the hybrid changed inside the box. Figure: check_set_holes.png.

| # | drawing | omission | hole | raw new px of type (largest) | kept px | hybrid changed (class: px) | why nothing / what |
|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | room (INK+detector) | 79 (43) | 204 | wall: 79, kitchen: 57, bedroom: 45, appliance: 28, door: 18, bathroom: 5 | room kept (204 px) on the omitted spot |
| 2 | colorful-1148 | door (a) | door (C8+detector) | 105 (105) | 39 | door: 105 | door kept (39 px) on the omitted spot |
| 4 | colorful-13689 | door (b) | door (detector) | 0 (0) | 0 | wall: 3 | the hybrid changed pixels but drew no door; the restriction reverts them all |
| 6 | colorful-13804 | window | window (detector) | 27 (27) | 27 | window: 27, wall: 1, living: 1 | window kept (27 px) on the omitted spot |
| 7 | colorful-13820 | wall | room (detector) | 13 (8) | 13 | wall: 13 | room kept (13 px) on the omitted spot |
| 8 | colorful-13824 | wall | wall (detector) | 6 (6) | 6 | wall: 6 | wall kept (6 px) on the omitted spot |
| 9 | colorful-13826 | fixture | sink (detector) | 144 (144) | 144 | sink: 144, bathroom: 7, other_room: 6 | sink kept (144 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 19 (19) | 23 | wall: 19, kitchen: 4 | room kept (23 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 9 (6) | 26 | bedroom: 14, wall: 9, living: 3 | room kept (26 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 44 (44) | 48 | wall: 44, kitchen: 4 | room kept (48 px) on the omitted spot |

## My verdicts (final CHECK with ink tier 2 off; overnight 2026-10-20)

| # | omission | verdict | why |
|---|---|---|---|
| 1 | door (a) | partly | found; the door hole merged into a "room" hole that straightened the angled wall and redrew room types |
| 2 | door (a) | correct | C8 + detector; one median door on the right wall |
| 3 | door (b) | wrong | R5 at another gap, a door added there |
| 4 | door (b) | wrong | found by the detector, but the parse already shows a door there; nothing added |
| 5 | window | wrong | nothing flagged |
| 6 | window | correct | window drawn on the erased spot |
| 7 | wall | wrong | 3 false ink doors (tier 1: this drawing's bedroom / bathroom doors have no arcs either; tier 2 off removed 3 more) |
| 8 | wall | correct | wall gap closed |
| 9 | fixture | partly | sink on the erased spot; a false appliance |
| 10 | fixture | wrong | nothing flagged |
| 11 | room | partly | walls redrawn in two holes on the erased band; one wall across an open area |
| 12 | intact | correct | nothing added |

**Score: 4 / 12 correct, 3 partly** (automatic 5 / 12), same as before ink tier 2 was switched off.

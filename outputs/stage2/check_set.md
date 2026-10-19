# Check set: 2 / 12 correct (fast mode, CPU)

found = a hole overlaps the omission; restored = the expected class was added there; extra = added components elsewhere; correct = restored and no extra (intact: nothing added).

| # | drawing | omission | found | restored | extra | correct | holes | s |
|---|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | no | no | 1 | no | 1 | 33 |
| 2 | colorful-1148 | door (a) | yes | no | 0 | no | 1 | 29 |
| 3 | colorful-1261 | door (b) | yes | no | 3 | no | 4 | 104 |
| 4 | colorful-13689 | door (b) | no | no | 0 | no | 0 | 2 |
| 5 | colorful-13513 | window | no | no | 0 | no | 0 | 2 |
| 6 | colorful-13804 | window | no | no | 0 | no | 0 | 2 |
| 7 | colorful-13820 | wall | no | no | 1 | no | 2 | 53 |
| 8 | colorful-13824 | wall | no | no | 0 | no | 0 | 5 |
| 9 | colorful-13826 | fixture | yes | yes | 0 | **yes** | 2 | 53 |
| 10 | colorful-14036 | fixture | no | no | 0 | no | 1 | 24 |
| 11 | colorful-14126 | room | yes | no | 1 | no | 3 | 72 |
| 12 | colorful-14147 | intact | no | no | 0 | **yes** | 0 | 2 |

## 1. colorful-11261 (door (a)): automatic verdict wrong

Removed: door (a). Detected: appliance (detector). Found no, restored no, extra 1.

Added (messages):

- appliance added in kitchen

## 2. colorful-1148 (door (a)): automatic verdict wrong

Removed: door (a). Detected: door (C8). Found yes, restored no, extra 0.

Added (messages):

- door missing (C8) -> nothing added (the model kept the drawing as it was)

## 3. colorful-1261 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (R5), railing (detector), sink (detector), sink (detector). Found yes, restored no, extra 3.

Added (messages):

- door missing (R5) -> nothing added (the model kept the drawing as it was)
- railing added in the plan
- sink added in bathroom

## 4. colorful-13689 (door (b)): automatic verdict wrong

Removed: door (b). Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 5. colorful-13513 (window): automatic verdict wrong

Removed: window. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 6. colorful-13804 (window): automatic verdict wrong

Removed: window. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 7. colorful-13820 (wall): automatic verdict wrong

Removed: wall. Detected: wall (detector), railing (detector). Found no, restored no, extra 1.

Added (messages):

- wall missing (detector) -> nothing added (the model kept the drawing as it was)
- railing added in the plan

## 8. colorful-13824 (wall): automatic verdict wrong

Removed: wall. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 9. colorful-13826 (fixture): automatic verdict correct

Removed: fixture. Detected: sink (detector), railing (detector). Found yes, restored yes, extra 0.

Added (messages):

- sink added in other_room
- railing missing (detector) -> nothing added (the model kept the drawing as it was)

## 10. colorful-14036 (fixture): automatic verdict wrong

Removed: fixture. Detected: door (R5). Found no, restored no, extra 0.

Added (messages):

- door missing (R5) -> nothing added (the model kept the drawing as it was)

## 11. colorful-14126 (room): automatic verdict wrong

Removed: room. Detected: door (R5), appliance (detector), railing (detector). Found yes, restored no, extra 1.

Added (messages):

- door missing (R5) -> nothing added (the model kept the drawing as it was)
- appliance missing (detector) -> nothing added (the model kept the drawing as it was)
- railing added in the plan

## 12. colorful-14147 (intact): automatic verdict correct

Removed: intact. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## Holes in the right place: raw hybrid output (step 3)

Per hole overlapping the omission: new pixels of the hole's type in the RAW hybrid output (before the type restriction), its largest new component, what the restriction kept, and the classes the hybrid changed inside the box. Figure: check_set_holes.png.

| # | drawing | omission | hole | raw new px of type (largest) | kept px | hybrid changed (class: px) | why nothing / what |
|---|---|---|---|---|---|---|---|
| 2 | colorful-1148 | door (a) | door (C8) | 0 (0) | 0 | other_room: 30 | the hybrid changed pixels but drew no door; the restriction reverts them all |
| 3 | colorful-1261 | door (b) | door (R5) | 0 (0) | 0 | other_room: 28 | the hybrid changed pixels but drew no door; the restriction reverts them all |
| 9 | colorful-13826 | fixture | sink (detector) | 70 (70) | 70 | sink: 70 | sink kept (70 px) on the omitted spot |
| 11 | colorful-14126 | room | appliance (detector) | 0 (0) | 0 | kitchen: 256, living: 199, bedroom: 81, wall: 18 | the hybrid changed pixels but drew no appliance; the restriction reverts them all |

## My verdicts (erasure v2 run, after looking at check_set.png and check_set_holes.png)

| # | omission | verdict | why |
|---|---|---|---|
| 1 | door (a) | wrong | doorway not flagged; a false appliance hole in the kitchen, an appliance added |
| 2 | door (a) | partly | C8 flagged the closed wall in the right place; the hybrid drew no door there (wall kept) |
| 3 | door (b) | partly | R5 fired next to the open doorway (box overlaps the omission's edge); no door drawn; a false railing and sink added |
| 4 | door (b) | wrong | nothing flagged (the parser labels the open gap as a door again) |
| 5 | window | wrong | nothing flagged |
| 6 | window | wrong | nothing flagged |
| 7 | wall | wrong | a wall hole in another place, nothing added there; a false railing added |
| 8 | wall | wrong | nothing flagged |
| 9 | fixture | correct | the sink hole was found and a sink was drawn on the erased spot; nothing else added |
| 10 | fixture | wrong | not flagged; a false R5 door hole (nothing added) |
| 11 | room | wrong | R5 / appliance / railing holes; nothing useful added; a false railing added |
| 12 | intact | correct | nothing flagged, nothing added |

**Score: 2 / 12 correct, 2 partly (right place found, nothing added).** Overnight (erasure v1): 1 / 12, 2 partly.
Settings were not changed based on this set.

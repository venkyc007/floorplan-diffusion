# Check set_v3: 6 / 12 correct (new definition), 5 / 12 (old) (fast mode, CPU)

Old: found = a hole overlaps the omission; restored = the expected class was added there; extra = added components elsewhere; correct = restored and no extra (intact: nothing added). New (user, 2026-10-20): fixed = a flag there and the final plan has the right class at the omission with IoU >= 0.3, whoever drew it; false adds = added or marked ('symbol missing in the drawing') components elsewhere; correct = fixed and no false adds.

| # | drawing | omission | found | restored | extra | correct (old) | fixed (IoU) | false adds | correct (new) | holes | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | yes | yes | 1 | no | no (0.04) | 1 | no | 1 | 32 |
| 2 | colorful-1148 | door (a) | yes | yes | 0 | **yes** | yes (0.80) | 0 | **yes** | 1 | 29 |
| 3 | colorful-1261 | door (b) | no | no | 2 | no | no (0.01) | 2 | no | 1 | 46 |
| 4 | colorful-13689 | door (b) | yes | no | 0 | no | yes (0.67) | 0 | **yes** | 1 | 4 |
| 5 | colorful-13513 | window | no | no | 0 | no | no (0.46) | 0 | no | 0 | 2 |
| 6 | colorful-13804 | window | yes | no | 0 | no | yes (0.71) | 0 | **yes** | 1 | 2 |
| 7 | colorful-13820 | wall | yes | yes | 0 | **yes** | no (0.18) | 4 | no | 5 | 31 |
| 8 | colorful-13824 | wall | yes | yes | 0 | **yes** | yes (0.57) | 0 | **yes** | 1 | 33 |
| 9 | colorful-13826 | fixture | yes | yes | 1 | no | yes (0.55) | 1 | no | 2 | 57 |
| 10 | colorful-14036 | fixture | no | no | 0 | no | no (0.00) | 0 | no | 0 | 1 |
| 11 | colorful-14126 | room | yes | yes | 0 | **yes** | yes (0.37) | 0 | **yes** | 3 | 71 |
| 12 | colorful-14147 | intact | no | no | 0 | **yes** | no (0.00) | 0 | **yes** | 0 | 4 |

## 1. colorful-11261 (door (a)): automatic verdict wrong

Removed: door (a). Detected: room (INK+detector). Found yes, restored yes, extra 1.

Added (messages):

- door added on the bedroom / hallway wall
- wall added next to hallway
- wall added next to hallway
- wall added next to hallway
- missing room -> room drawn as kitchen

## 2. colorful-1148 (door (a)): automatic verdict correct

Removed: door (a). Detected: door (C8+detector). Found yes, restored yes, extra 0.

Added (messages):

- bedroom had no door -> door added on the bedroom / hallway wall

## 3. colorful-1261 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (R5). Found no, restored no, extra 2.

Added (messages):

- open doorway between hallway and other_room had no door -> door added
- wall added next to other_room

## 4. colorful-13689 (door (b)): automatic verdict wrong

Removed: door (b). Detected: door (detector). Found yes, restored no, extra 0.

Added (messages):

- door symbol missing in the drawing -> added

## 5. colorful-13513 (window): automatic verdict wrong

Removed: window. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## 6. colorful-13804 (window): automatic verdict wrong

Removed: window. Detected: window (detector). Found yes, restored no, extra 0.

Added (messages):

- window symbol missing in the drawing -> added

## 7. colorful-13820 (wall): automatic verdict correct

Removed: wall. Detected: door (detector), room (detector), door (INK), door (INK), door (INK). Found yes, restored yes, extra 0.

Added (messages):

- door symbol missing in the drawing -> added
- wall added next to other_room
- wall added next to other_room

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
- missing room -> room drawn as bedroom
- wall added next to living

## 12. colorful-14147 (intact): automatic verdict correct

Removed: intact. Detected: nothing. Found no, restored no, extra 0.

Added (messages):

- (nothing)

## Holes in the right place: raw hybrid output (step 3)

Per hole overlapping the omission: new pixels of the hole's type in the RAW hybrid output (before the type restriction), its largest new component, what the restriction kept, and the classes the hybrid changed inside the box. Figure: check_set_holes_v3.png.

| # | drawing | omission | hole | raw new px of type (largest) | kept px | hybrid changed (class: px) | why nothing / what |
|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | room (INK+detector) | 79 (43) | 154 | wall: 79, kitchen: 57, bedroom: 45, appliance: 28, door: 18, bathroom: 5 | room kept (154 px) on the omitted spot |
| 2 | colorful-1148 | door (a) | door (C8+detector) | 54 (54) | 39 | door: 54 | door kept (39 px) on the omitted spot |
| 7 | colorful-13820 | wall | room (detector) | 17 (9) | 17 | wall: 17 | room kept (17 px) on the omitted spot |
| 8 | colorful-13824 | wall | wall (detector) | 6 (6) | 6 | wall: 6 | wall kept (6 px) on the omitted spot |
| 9 | colorful-13826 | fixture | sink (detector) | 146 (146) | 146 | sink: 146, bathroom: 6, other_room: 6 | sink kept (146 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 19 (19) | 23 | wall: 19, kitchen: 4 | room kept (23 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 9 (6) | 23 | bedroom: 14, wall: 9, living: 3 | room kept (23 px) on the omitted spot |
| 11 | colorful-14126 | room | room (detector) | 44 (44) | 48 | wall: 44, kitchen: 4 | room kept (48 px) on the omitted spot |

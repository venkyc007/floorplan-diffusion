# Check set: 1 / 12 correct (fast mode, CPU)

found = a hole overlaps the omission; restored = the expected class was added there; extra = added components elsewhere; correct = restored and no extra (intact: nothing added).

| # | drawing | omission | found | restored | extra | correct | holes | s |
|---|---|---|---|---|---|---|---|---|
| 1 | colorful-11261 | door (a) | no | no | 1 | no | 1 | 34 |
| 2 | colorful-1148 | door (a) | yes | no | 0 | no | 1 | 29 |
| 3 | colorful-1261 | door (b) | no | no | 3 | no | 4 | 108 |
| 4 | colorful-13689 | door (b) | no | no | 0 | no | 0 | 2 |
| 5 | colorful-13513 | window | no | no | 0 | no | 0 | 2 |
| 6 | colorful-13804 | window | no | no | 0 | no | 0 | 2 |
| 7 | colorful-13820 | wall | no | no | 1 | no | 2 | 53 |
| 8 | colorful-13824 | wall | no | no | 0 | no | 0 | 4 |
| 9 | colorful-13826 | fixture | no | no | 0 | no | 1 | 30 |
| 10 | colorful-14036 | fixture | no | no | 0 | no | 1 | 25 |
| 11 | colorful-14126 | room | yes | no | 1 | no | 3 | 76 |
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

Removed: door (b). Detected: door (R5), railing (detector), sink (detector), sink (detector). Found no, restored no, extra 3.

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

## 9. colorful-13826 (fixture): automatic verdict wrong

Removed: fixture. Detected: railing (detector). Found no, restored no, extra 0.

Added (messages):

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


## My verdicts (after looking at check_set.png)

| # | omission | verdict | why |
|---|---|---|---|
| 1 | door (a) | wrong | doorway not flagged; a false "appliance" hole in the kitchen and an appliance added there |
| 2 | door (a) | partly | C8 flagged the cut-off bedroom at the right wall, but the fill added nothing (the first run, with the older erasure, did add the door) |
| 3 | door (b) | partly | R5 fired at the open doorway on the upper floor (box next to it, not overlapping, so the automatic check says "not found"); nothing added there; 3 false holes, a railing and a sink added elsewhere |
| 4 | door (b) | wrong | the parser drew the erased doorway back as a door, so nothing was missing for CHECK |
| 5 | window | wrong | the parser drew the erased window back (the erased spot is also visibly blurred here); nothing flagged |
| 6 | window | wrong | as 5: window redrawn by the parser, nothing flagged |
| 7 | wall | wrong | a wall hole flagged in another building part; a false railing added |
| 8 | wall | wrong | nothing flagged |
| 9 | fixture | wrong | not flagged; a false railing hole (nothing added) |
| 10 | fixture | wrong | not flagged; a false R5 door hole (nothing added) |
| 11 | room | wrong | removed walls only partly gone in the image; R5 + appliance holes nearby, nothing useful added; a false railing added |
| 12 | intact | correct | nothing flagged, nothing added |

**Score: 1 / 12 correct, 2 partly (both: the right place found, nothing added).** Settings were not changed based on this set.
Main causes, in order: (1) the parser redraws erased doors and windows from context (see the image-level evaluation in
detection_eval.md: 55-85% of a removed door's pixels and 77% of a window's are labelled back), so CHECK has nothing to find;
(2) on parsed drawings the detector misses most real omissions and raises false railing / appliance holes; (3) when a hole is
found, the fast fill with the type restriction often adds nothing of that type.

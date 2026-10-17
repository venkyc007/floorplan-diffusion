# Gate G2: constraint violations on real plans

1000 ground-truth training windows (128 x 128, completion scale; seed 0) around random typed deletions (window 160, region 158, wall 157, fixture 151, door 142, room 134, railing 98), from 3944 plans. V computed on the ground truth in its absolute (inference) form inside M = hole grown 3 px; a window violates if the summed V in M is >= 1. C17 is GT-relative (0 on the ground truth). Rule: absolute mode violated in > 10% of windows -> GT-hinged; > 40% -> propose to drop (user decides).

| constraint | title | default mode | % windows violating | mean V in M | decision |
|---|---|---|---|---|---|
| C1 | walls connect (no dead ends) | gt_relative | 19.3 | 0.00065 | keep gt_relative |
| C2 | no floating wall pieces | absolute | 0.8 | 0.00050 | keep absolute |
| C3 | walls axis-aligned | gt_hinged | 10.7 | 0.00080 | keep gt_hinged |
| C4 | wall thickness in range | gt_hinged | 42.9 | 0.01422 | keep gt_hinged |
| C5 | walls continue into the hole | absolute | 14.6 | 0.00250 | switch to gt_hinged |
| C6 | different spaces never touch | absolute | 1.5 | 0.00036 | keep absolute |
| C7 | minimum width per space/fixture | gt_hinged | 40.5 | 0.03454 | keep gt_hinged |
| C8 | every room reachable through doors | gt_relative | 18.6 | 0.04590 | keep gt_relative |
| C9 | openings sit in a wall | absolute | 18.0 | 0.00754 | switch to gt_hinged |
| C10 | openings cut through the wall | gt_hinged | 15.9 | 0.00067 | keep gt_hinged |
| C11 | door joins two spaces | absolute | 18.8 | 0.00422 | switch to gt_hinged |
| C12 | window: inside one side, outside other | absolute | 23.9 | 0.01414 | switch to gt_hinged |
| C13 | opening length in range | gt_hinged | 4.7 | 0.00105 | keep gt_hinged |
| C14 | door clearance | gt_hinged | 9.4 | 0.00101 | keep gt_hinged |
| C15 | fixture in a compatible room | absolute | 1.9 | 0.00052 | keep absolute |
| C16 | fixture against a wall | gt_hinged | 1.9 | 0.00106 | keep gt_hinged |
| C17 | required fixtures | gt_relative | 0.0 | 0.00000 | keep gt_relative |
| C18 | railing borders outdoor | absolute | 8.2 | 0.00517 | keep absolute |

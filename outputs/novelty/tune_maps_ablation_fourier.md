# Large holes on the TUNING drawings, map level_ffc

120 erased versions of tuning val drawings (parser's map of the erased drawing; kinds: region 65, room (whole) 55). Hole given = the box of the true omission area; auto = the locator's flags, each filled by the new hybrid. Fast sampling. See src/novelty/eval_maps.py for the scores.

| arm | fixed | recovered (2 px) | missing-wall recall | added-wall precision | wall IoU | oracle wall IoU (best of N) | tol. mIoU (parse -> fill) | wall F1 (2 px) | valid | violations (truth) | found | s per hole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sketch | 38% | 47% | 0.57 | 0.62 | 0.267 | nan | 28.3 -> 33.9 | 0.752 | 3% | 33.1 (8.3) | given | 1.1 |
| hybrid@700 | 38% | 46% | 0.53 | 0.62 | 0.266 | 0.283 | 28.3 -> 33.5 | 0.740 | 16% | 17.7 (8.3) | given | 8.6 |

By kind (fixed / recovered / tol. mIoU / valid):

| kind | n | sketch | hybrid@700 |
|---|---|---|---|
| region | 65 | 45% / 42% / 37.7 / 3% | 45% / 45% / 37.0 / 17% |
| room (whole) | 55 | 31% / 53% / 29.5 / 4% | 29% / 47% / 29.2 / 15% |

By size bin (fixed / recovered / tol. mIoU / valid):

| size_bin | n | sketch | hybrid@700 |
|---|---|---|---|
| large | 81 | 36% / 48% / 35.3 / 4% | 37% / 46% / 34.7 / 17% |
| medium | 39 | 44% / 44% / 31.1 / 3% | 38% / 46% / 30.8 / 13% |

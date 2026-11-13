# Large holes on the TUNING drawings, map level_plain

120 erased versions of tuning val drawings (parser's map of the erased drawing; kinds: region 65, room (whole) 55). Hole given = the box of the true omission area; auto = the locator's flags, each filled by the new hybrid. Fast sampling. See src/novelty/eval_maps.py for the scores.

| arm | fixed | recovered (2 px) | missing-wall recall | added-wall precision | wall IoU | oracle wall IoU (best of N) | tol. mIoU (parse -> fill) | wall F1 (2 px) | valid | violations (truth) | found | s per hole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sketch | 40% | 45% | 0.57 | 0.61 | 0.267 | nan | 28.3 -> 32.9 | 0.747 | 4% | 33.8 (8.3) | given | 1.1 |
| hybrid@700 | 39% | 45% | 0.53 | 0.61 | 0.264 | 0.279 | 28.3 -> 32.1 | 0.733 | 15% | 17.8 (8.3) | given | 8.5 |

By kind (fixed / recovered / tol. mIoU / valid):

| kind | n | sketch | hybrid@700 |
|---|---|---|---|
| region | 65 | 45% / 42% / 36.4 / 5% | 45% / 42% / 35.8 / 18% |
| room (whole) | 55 | 35% / 49% / 28.7 / 4% | 33% / 49% / 27.7 / 11% |

By size bin (fixed / recovered / tol. mIoU / valid):

| size_bin | n | sketch | hybrid@700 |
|---|---|---|---|
| large | 81 | 38% / 43% / 34.4 / 5% | 38% / 44% / 33.6 / 16% |
| medium | 39 | 44% / 49% / 29.7 / 3% | 41% / 46% / 29.0 / 13% |

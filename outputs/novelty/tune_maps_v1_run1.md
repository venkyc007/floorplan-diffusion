# Large holes on the TUNING drawings, map level_ffc

120 erased versions of tuning val drawings (parser's map of the erased drawing; kinds: region 65, room (whole) 55). Hole given = the box of the true omission area; auto = the locator's flags, each filled by the new hybrid. Fast sampling. See src/novelty/eval_maps.py for the scores.

| arm | fixed | wall IoU | oracle wall IoU (best of N) | tol. mIoU (parse -> fill) | wall F1 (2 px) | valid | violations (truth) | found | s per hole |
|---|---|---|---|---|---|---|---|---|---|
| b4 | 40% | 0.265 | nan | 28.3 -> 29.9 | 0.680 | 18% | 17.6 (8.3) | given | 3.3 |
| sketch | 38% | 0.267 | nan | 28.3 -> 33.9 | 0.752 | 3% | 33.1 (8.3) | given | 1.8 |
| hybrid@500 | 38% | 0.267 | 0.276 | 28.3 -> 33.6 | 0.740 | 14% | 18.5 (8.3) | given | 8.6 |
| hybrid@700 | 38% | 0.266 | 0.283 | 28.3 -> 33.5 | 0.740 | 16% | 17.7 (8.3) | given | 8.7 |
| hybrid@900 | 38% | 0.271 | 0.296 | 28.3 -> 33.0 | 0.726 | 16% | 16.9 (8.3) | given | 8.7 |

By kind (fixed / tol. mIoU / valid):

| kind | n | b4 | sketch | hybrid@500 | hybrid@700 | hybrid@900 |
|---|---|---|---|---|---|---|
| region | 65 | 49% / 36.2 / 20% | 45% / 37.7 / 3% | 46% / 37.0 / 15% | 45% / 37.0 / 17% | 40% / 36.7 / 17% |
| room (whole) | 55 | 29% / 22.5 / 15% | 31% / 29.5 / 4% | 29% / 29.6 / 13% | 29% / 29.2 / 15% | 36% / 28.5 / 15% |

By size bin (fixed / tol. mIoU / valid):

| size_bin | n | b4 | sketch | hybrid@500 | hybrid@700 | hybrid@900 |
|---|---|---|---|---|---|---|
| large | 81 | 41% / 32.1 / 16% | 36% / 35.3 / 4% | 37% / 34.8 / 15% | 37% / 34.7 / 17% | 36% / 34.5 / 15% |
| medium | 39 | 38% / 25.5 / 21% | 44% / 31.1 / 3% | 41% / 31.1 / 13% | 38% / 30.8 / 13% | 44% / 29.7 / 18% |

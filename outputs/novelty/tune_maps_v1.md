# Large holes on the TUNING drawings, map level_ffc

120 erased versions of tuning val drawings (parser's map of the erased drawing; kinds: region 65, room (whole) 55). Hole given = the box of the true omission area; auto = the locator's flags, each filled by the new hybrid. Fast sampling. See src/novelty/eval_maps.py for the scores.

| arm | fixed | recovered (2 px) | missing-wall recall | added-wall precision | wall IoU | oracle wall IoU (best of N) | tol. mIoU (parse -> fill) | wall F1 (2 px) | valid | violations (truth) | found | s per hole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| truth pasted (ceiling) | 72% | 96% | 1.00 | 1.00 | 0.448 | nan | 28.3 -> 100.0 | 1.000 | 2% | 19.8 (8.3) | given | 0.0 |
| b4 | 40% | 37% | 0.46 | 0.56 | 0.265 | nan | 28.3 -> 29.9 | 0.680 | 18% | 17.6 (8.3) | given | 1.9 |
| sketch | 38% | 47% | 0.57 | 0.62 | 0.267 | nan | 28.3 -> 33.9 | 0.752 | 3% | 33.1 (8.3) | given | 0.5 |
| hybrid@700 | 38% | 46% | 0.53 | 0.62 | 0.266 | 0.283 | 28.3 -> 33.5 | 0.740 | 16% | 17.7 (8.3) | given | 8.9 |
| hybrid@900 | 38% | 48% | 0.53 | 0.58 | 0.271 | 0.296 | 28.3 -> 33.0 | 0.726 | 16% | 16.9 (8.3) | given | 9.0 |
| auto | 22% | 29% | 0.35 | 0.67 | 0.205 | nan | 28.3 -> 31.9 | 0.679 | 15% | 18.1 (8.3) | 66% | 3.6 |

By kind (fixed / recovered / tol. mIoU / valid):

| kind | n | truth pasted (ceiling) | b4 | sketch | hybrid@700 | hybrid@900 | auto |
|---|---|---|---|---|---|---|---|
| region | 65 | 60% / 92% / 100.0 / 2% | 49% / 38% / 36.2 / 20% | 45% / 42% / 37.7 / 3% | 45% / 45% / 37.0 / 17% | 40% / 43% / 36.7 / 17% | 18% / 23% / 36.2 / 14% |
| room (whole) | 55 | 87% / 100% / 100.0 / 2% | 29% / 35% / 22.5 / 15% | 31% / 53% / 29.5 / 4% | 29% / 47% / 29.2 / 15% | 36% / 53% / 28.5 / 15% | 25% / 36% / 26.7 / 16% |

By size bin (fixed / recovered / tol. mIoU / valid):

| size_bin | n | truth pasted (ceiling) | b4 | sketch | hybrid@700 | hybrid@900 | auto |
|---|---|---|---|---|---|---|---|
| large | 81 | 69% / 94% / 100.0 / 1% | 41% / 37% / 32.1 / 16% | 36% / 48% / 35.3 / 4% | 37% / 46% / 34.7 / 17% | 36% / 46% / 34.5 / 15% | 17% / 27% / 34.7 / 16% |
| medium | 39 | 79% / 100% / 100.0 / 3% | 38% / 36% / 25.5 / 21% | 44% / 44% / 31.1 / 3% | 38% / 46% / 30.8 / 13% | 44% / 51% / 29.7 / 18% | 31% / 33% / 26.0 / 13% |

Intact drawings (100; auto): region flags per drawing 0.08, drawings with a flag 8%, added components per drawing 0.35.

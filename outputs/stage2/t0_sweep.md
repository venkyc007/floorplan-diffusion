# U-Net sketch alone vs hybrid at several t0 (CPU)

The same 30 val holes as fast_vs_full.md (fixed windows, seed 7; kinds: door 4, fixture 6, railing 2, region 7, room 5, wall 1, window 5). Same t0 for every hole. Ground-truth windows: 3.53 violations per hole on average. Means per hole.

| row | strict mIoU | tolerant mIoU | validity % | violations / hole | walls | openings | spaces | fixtures |
|---|---|---|---|---|---|---|---|---|
| unet | 63.4 | 67.1 | 27 | 10.70 | 5.17 | 2.17 | 3.27 | 0.10 |
| fast200 | 65.9 | 69.5 | 73 | 3.80 | 1.37 | 0.60 | 1.70 | 0.13 |
| fast300 | 65.7 | 69.2 | 77 | 3.70 | 1.37 | 0.60 | 1.67 | 0.07 |
| fast400 | 66.1 | 69.7 | 77 | 3.77 | 1.40 | 0.57 | 1.73 | 0.07 |
| fast600 | 65.9 | 69.5 | 73 | 3.30 | 1.03 | 0.53 | 1.67 | 0.07 |

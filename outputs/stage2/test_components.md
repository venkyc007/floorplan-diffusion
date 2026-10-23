# Stage 2 TEST, second evaluation: end to end on component-level omissions (all test drawings)

Code version: final 7581cb2 + uncommitted changes. Frozen settings; fast sampling (as validated). One single-component omission per test drawing, kinds rotated (src/stage2/test_components.py). Brackets: Wilson 95% intervals.

Drawings: 371 processed, 371 scored, 0 skipped (no component of any kind), 0 errors.

| cases | n | found | right type | fixed (any source) | fixed by the fill | right class present (flag or not) | false additions per drawing | tol. mIoU in the region | s per drawing |
|---|---|---|---|---|---|---|---|---|---|
| all | 371 | 62% [57, 67] | 51% [46, 56] | 41% [36, 46] | 22% [18, 27] | 51% [46, 56] | 0.92 | 59.2 -> 62.0 | 3 |
| all except railings (manual-only by design) | 330 | 69% [64, 74] | 57% [52, 62] | 46% [41, 52] | 25% [21, 30] | 56% [50, 61] | 0.92 | 59.0 -> 62.1 | 3 |

## By omission kind

| cases | n | found | right type | fixed (any source) | fixed by the fill | right class present (flag or not) | false additions per drawing | tol. mIoU in the region | s per drawing |
|---|---|---|---|---|---|---|---|---|---|
| door (a) | 54 | 65% [51, 76] | 52% [39, 65] | 37% [25, 50] | 17% [9, 29] | 44% [32, 58] | 0.74 | 56.1 -> 59.8 | 3 |
| door (b) | 52 | 56% [42, 68] | 48% [35, 61] | 46% [33, 59] | 0% [0, 7] | 81% [68, 89] | 0.79 | 75.7 -> 75.3 | 3 |
| window | 53 | 68% [55, 79] | 68% [55, 79] | 62% [49, 74] | 21% [12, 33] | 68% [55, 79] | 1.09 | 66.6 -> 71.4 | 4 |
| wall | 54 | 80% [67, 88] | 46% [34, 59] | 57% [44, 70] | 43% [30, 56] | 59% [46, 71] | 0.70 | 51.9 -> 58.6 | 3 |
| railing | 41 | 2% [0, 13] | 0% [0, 9] | 0% [0, 9] | 0% [0, 9] | 15% [7, 28] | 0.95 | 61.1 -> 61.0 | 3 |
| fixture | 64 | 55% [43, 66] | 41% [29, 53] | 17% [10, 28] | 19% [11, 30] | 22% [14, 33] | 1.05 | 55.3 -> 57.3 | 3 |
| room | 53 | 94% [85, 98] | 92% [82, 97] | 64% [51, 76] | 53% [40, 66] | 68% [55, 79] | 1.09 | 49.5 -> 51.6 | 4 |

By type:

| type | n | found | right type | fixed | fixed by the fill | false adds |
|---|---|---|---|---|---|---|
| door (a) | 54 | 65% | 52% | 37% | 17% | 0.74 |
| door (b) | 52 | 56% | 48% | 46% | 0% | 0.79 |
| window | 53 | 68% | 68% | 62% | 21% | 1.09 |
| wall | 54 | 80% | 46% | 57% | 43% | 0.70 |
| railing | 41 | 2% | 0% | 0% | 0% | 0.95 |
| fixture | 64 | 55% | 41% | 17% | 19% | 1.05 |
| room | 53 | 94% | 92% | 64% | 53% | 1.09 |

By location:

| location | n | found | right type | fixed | fixed by the fill | false adds |
|---|---|---|---|---|---|---|
| interior | 183 | 58% | 47% | 33% | 21% | 0.94 |
| exterior | 188 | 65% | 55% | 49% | 24% | 0.90 |

By position:

| position | n | found | right type | fixed | fixed by the fill | false adds |
|---|---|---|---|---|---|---|
| centre | 227 | 65% | 55% | 41% | 25% | 1.04 |
| edge | 144 | 56% | 45% | 42% | 18% | 0.73 |

By size_bin:

| size_bin | n | found | right type | fixed | fixed by the fill | false adds |
|---|---|---|---|---|---|---|
| small | 367 | 62% | 51% | 41% | 22% | 0.92 |
| medium | 4 | 50% | 25% | 25% | 25% | 0.75 |

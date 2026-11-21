# Round 4 on the 208 tuning drawings: accuracy and structural soundness by pipeline version (kernels v14, v15)

End to end, nothing given (the omission is found by the pipeline). Rows 1-6: the same pipeline with one more use of the constraints switched on (kernel v14); v4c and v4c-keep: added after v14 (kernel v15, same cases). Tuning drawings: used to choose the final version; optimistic by construction.

Soundness = the three structural checks of the review on the omission's region (src/novelty/soundness.py), judged without the true layout. Rule check = the project's 18-rule check (violation blobs in a window around the omission), in four groups; 'valid vs oracle' = no more violations than the parser's map with the true layout pasted into the erased area.

## large omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net sketch alone | 208 | 96% | 81% | 57% | 43.0 | 59% | 95% | 50% | 27% | 2.94 | 43.2 | 23.2 | 4.0 | 15.7 | 0.3 | 40% |
| + diffusion finish, 1 sample | 208 | 96% | 79% | 58% | 42.2 | 58% | 93% | 45% | 25% | 1.69 | 24.9 | 12.4 | 2.6 | 9.4 | 0.6 | 87% |
| v3c (+ best of 2 by rule count) | 208 | 96% | 81% | 57% | 42.4 | 55% | 93% | 45% | 25% | 1.74 | 24.1 | 11.9 | 2.5 | 9.1 | 0.5 | 87% |
| v3d (+ clean-up) | 208 | 96% | 66% | 47% | 40.8 | 65% | 78% | 46% | 31% | 1.44 | 26.9 | 12.2 | 2.7 | 11.5 | 0.4 | 85% |
| v4 (+ repair) | 208 | 96% | 64% | 48% | 40.0 | 87% | 94% | 88% | 75% | 1.23 | 25.9 | 11.7 | 3.4 | 10.2 | 0.6 | 85% |
| v4b (+ selection by soundness) | 208 | 96% | 68% | 47% | 40.1 | 89% | 96% | 90% | 79% | 1.00 | 25.6 | 11.8 | 3.3 | 10.0 | 0.6 | 87% |
| v4c (v4b, no wall-end rule) | 208 | 96% | 75% | 48% | 40.8 | 80% | 97% | 90% | 71% | 1.14 | 25.3 | 12.2 | 2.8 | 9.7 | 0.6 | 88% |
| v4c-keep (v4c, repair never deletes walls) | 208 | 96% | 75% | 45% | 41.3 | 79% | 96% | 89% | 69% | 1.14 | 26.3 | 12.4 | 3.4 | 10.0 | 0.6 | 87% |
| *parser's map given (nothing filled)* | 208 | - | - | - | - | 44% | 72% | 91% | 32% | 3.88 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 208 | - | - | - | - | 64% | 80% | 91% | 48% | 3.67 | 36.7 | 16.5 | 5.0 | 13.9 | 1.3 | - |
| *true layout* | 208 | - | - | - | - | 79% | 97% | 96% | 74% | 0.67 | 10.5 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| U-Net sketch alone | 22% | 5% | 73% |
| + diffusion finish, 1 sample | 21% | 4% | 75% |
| v3c (+ best of 2 by rule count) | 20% | 5% | 75% |
| v3d (+ clean-up) | 22% | 9% | 69% |
| v4 (+ repair) | 52% | 22% | 25% |
| v4b (+ selection by soundness) | 59% | 21% | 21% |
| v4c (v4b, no wall-end rule) | 56% | 15% | 29% |
| v4c-keep (v4c, repair never deletes walls) | 55% | 14% | 31% |

Paired, v4b (+ selection by soundness) minus v3c (+ best of 2 by rule count) (large): recovered -13.5 [-18.3, -8.7], fixed -10.6 [-15.9, -5.8], sound +54.8 [+47.6, +61.5], rule violations +1.6 [+0.4, +2.8].

Paired, v4c (v4b, no wall-end rule) minus v4b (+ selection by soundness) (large): recovered +7.7 [+3.4, +12.0], fixed +1.0 [-1.4, +3.4], sound -8.2 [-12.5, -3.8], rule violations -0.3 [-0.8, +0.1].

Paired, v4c-keep (v4c, repair never deletes walls) minus v4b (+ selection by soundness) (large): recovered +7.2 [+3.4, +11.1], fixed -1.4 [-4.8, +1.4], sound -10.1 [-14.9, -5.8], rule violations +0.7 [+0.1, +1.2].

Paired, v4c-keep (v4c, repair never deletes walls) minus v3c (+ best of 2 by rule count) (large): recovered -6.2 [-10.1, -2.4], fixed -12.0 [-17.3, -6.7], sound +44.7 [+37.5, +51.9], rule violations +2.2 [+1.1, +3.4].

Paired, v4b (+ selection by soundness) minus v4 (+ repair) (large): recovered +3.8 [+0.0, +7.7], fixed -1.0 [-3.4, +1.4], sound +4.8 [+1.9, +8.2], rule violations -0.3 [-0.9, +0.3].

Paired, + diffusion finish, 1 sample minus U-Net sketch alone (large): recovered -1.9 [-3.8, -0.5], fixed +1.4 [-0.5, +3.8], sound -1.9 [-5.8, +1.9], rule violations -18.3 [-20.5, -16.2].

## medium omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net sketch alone | 208 | 89% | 72% | 50% | 43.3 | 72% | 91% | 54% | 33% | 1.71 | 29.5 | 15.8 | 2.2 | 11.3 | 0.2 | 50% |
| + diffusion finish, 1 sample | 208 | 89% | 73% | 51% | 42.1 | 73% | 91% | 55% | 35% | 0.92 | 17.7 | 8.5 | 1.6 | 7.1 | 0.4 | 87% |
| v3c (+ best of 2 by rule count) | 208 | 89% | 73% | 51% | 42.4 | 73% | 92% | 54% | 35% | 0.89 | 16.9 | 8.2 | 1.4 | 6.9 | 0.4 | 89% |
| v3d (+ clean-up) | 208 | 89% | 63% | 43% | 41.0 | 77% | 85% | 63% | 45% | 0.92 | 20.4 | 9.6 | 1.5 | 8.9 | 0.3 | 84% |
| v4 (+ repair) | 208 | 89% | 61% | 40% | 40.9 | 89% | 92% | 93% | 78% | 0.80 | 20.0 | 9.3 | 2.0 | 8.1 | 0.6 | 85% |
| v4b (+ selection by soundness) | 208 | 89% | 62% | 40% | 40.8 | 91% | 93% | 94% | 82% | 0.68 | 20.4 | 9.5 | 2.1 | 8.2 | 0.7 | 83% |
| v4c (v4b, no wall-end rule) | 208 | 89% | 65% | 41% | 41.2 | 89% | 93% | 95% | 81% | 0.74 | 20.3 | 9.7 | 1.9 | 8.1 | 0.6 | 82% |
| v4c-keep (v4c, repair never deletes walls) | 208 | 89% | 66% | 41% | 41.3 | 86% | 93% | 95% | 77% | 0.70 | 20.5 | 9.8 | 1.9 | 8.2 | 0.6 | 82% |
| *parser's map given (nothing filled)* | 208 | - | - | - | - | 49% | 67% | 93% | 33% | 2.70 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 208 | - | - | - | - | 71% | 81% | 92% | 53% | 2.42 | 27.4 | 12.5 | 3.2 | 10.6 | 1.0 | - |
| *true layout* | 208 | - | - | - | - | 88% | 97% | 98% | 84% | 0.45 | 7.6 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| U-Net sketch alone | 19% | 14% | 67% |
| + diffusion finish, 1 sample | 22% | 13% | 65% |
| v3c (+ best of 2 by rule count) | 22% | 12% | 65% |
| v3d (+ clean-up) | 26% | 18% | 55% |
| v4 (+ repair) | 51% | 27% | 22% |
| v4b (+ selection by soundness) | 56% | 26% | 18% |
| v4c (v4b, no wall-end rule) | 56% | 25% | 19% |
| v4c-keep (v4c, repair never deletes walls) | 55% | 23% | 23% |

Paired, v4b (+ selection by soundness) minus v3c (+ best of 2 by rule count) (medium): recovered -10.1 [-14.9, -5.8], fixed -10.6 [-15.4, -6.7], sound +47.6 [+40.9, +54.3], rule violations +3.5 [+2.6, +4.4].

Paired, v4c (v4b, no wall-end rule) minus v4b (+ selection by soundness) (medium): recovered +2.4 [-1.4, +6.7], fixed +0.5 [-2.4, +3.4], sound -1.4 [-3.8, +0.5], rule violations -0.0 [-0.4, +0.3].

Paired, v4c-keep (v4c, repair never deletes walls) minus v4b (+ selection by soundness) (medium): recovered +3.8 [-0.5, +8.2], fixed +1.0 [-2.4, +4.3], sound -4.8 [-8.2, -1.9], rule violations +0.1 [-0.2, +0.5].

Paired, v4c-keep (v4c, repair never deletes walls) minus v3c (+ best of 2 by rule count) (medium): recovered -6.2 [-10.6, -2.4], fixed -9.6 [-13.9, -5.8], sound +42.8 [+36.1, +49.5], rule violations +3.6 [+2.7, +4.5].

Paired, v4b (+ selection by soundness) minus v4 (+ repair) (medium): recovered +1.4 [-1.0, +3.8], fixed +0.0 [-2.9, +2.9], sound +3.8 [+1.0, +6.7], rule violations +0.3 [-0.1, +0.8].

Paired, + diffusion finish, 1 sample minus U-Net sketch alone (medium): recovered +1.0 [-1.0, +2.9], fixed +1.4 [-1.0, +3.8], sound +1.4 [-2.4, +5.3], rule violations -11.8 [-13.5, -10.1].

## intact drawings (nothing erased)

| version | n | drawings with a region flag | added components / drawing |
|---|---|---|---|
| U-Net sketch alone | 208 | 8.7% | 1.38 |
| + diffusion finish, 1 sample | 208 | 8.7% | 1.23 |
| v3c (+ best of 2 by rule count) | 208 | 8.7% | 1.27 |
| v3d (+ clean-up) | 208 | 8.7% | 0.89 |
| v4 (+ repair) | 208 | 8.7% | 0.95 |
| v4b (+ selection by soundness) | 208 | 8.7% | 0.93 |
| v4c (v4b, no wall-end rule) | 208 | 8.7% | 0.93 |
| v4c-keep (v4c, repair never deletes walls) | 208 | 8.7% | 0.93 |

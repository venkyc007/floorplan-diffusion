# Round 4 TEST result: all versions on the CubiCasa5K test drawings (v4c: kernel v17; v3c, v4b, shipped: kernel v18; each run once)

End to end, nothing given. Every typed test drawing (parser_test.zip) except the check set; one small, one medium and one large omission per drawing, plus the intact drawing; the same cases for every version (same generator and seeds). All versions frozen before their test run (chosen on tuning val, checked on held-out val). v4c was run first (kernel v17, user-approved 2026-11-21) and stays the final hybrid; v3c, v4b and the shipped Stage 2 pipeline were added on the user's request the same day (kernel v18) for the comparison only; nothing was chosen or changed on these results. Rows: outputs/novelty/e2e_test_all_arms.csv.

Soundness = the three structural checks of the review on the omission's region (src/novelty/soundness.py), judged without the true layout. Rule check = the project's 18-rule check (violation blobs in a window around the omission), in four groups; 'valid vs oracle' = no more violations than the parser's map with the true layout pasted into the erased area.

## large omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped Stage 2 | 369 | 82% | 8% | 10% | 37.3 | 52% | 75% | 86% | 36% | 3.84 | 32.8 | 16.2 | 4.3 | 11.6 | 0.5 | 79% |
| v3c (round 2 final, accuracy first) | 369 | 98% | 78% | 56% | 42.6 | 59% | 93% | 35% | 19% | 2.01 | 25.6 | 12.1 | 2.4 | 10.5 | 0.6 | 90% |
| v4b (most sound on tuning) | 369 | 98% | 68% | 49% | 40.2 | 95% | 96% | 92% | 85% | 1.25 | 27.7 | 12.3 | 3.2 | 11.4 | 0.8 | 87% |
| v4c (round 4 final) | 369 | 98% | 66% | 47% | 40.6 | 86% | 96% | 92% | 78% | 1.38 | 27.6 | 12.9 | 2.9 | 11.1 | 0.7 | 87% |
| *parser's map given (nothing filled)* | 369 | - | - | - | - | 48% | 66% | 90% | 31% | 4.39 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 369 | - | - | - | - | 59% | 78% | 89% | 44% | 4.12 | 39.9 | 18.2 | 5.4 | 15.0 | 1.3 | - |
| *true layout* | 369 | - | - | - | - | 77% | 93% | 96% | 68% | 0.71 | 11.3 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| shipped Stage 2 | 2% | 33% | 64% |
| v3c (round 2 final, accuracy first) | 13% | 6% | 81% |
| v4b (most sound on tuning) | 59% | 25% | 15% |
| v4c (round 4 final) | 54% | 24% | 22% |

Paired, v4c (round 4 final) minus v3c (round 2 final, accuracy first) (large): recovered -11.9 [-16.0, -8.4], fixed -8.1 [-11.7, -4.3], sound +59.3 [+53.9, +64.5], rule violations +2.1 [+1.2, +2.9].

Paired, v4c (round 4 final) minus v4b (most sound on tuning) (large): recovered -1.6 [-4.9, +1.4], fixed -1.1 [-3.3, +1.1], sound -6.2 [-9.2, -3.5], rule violations -0.1 [-0.4, +0.2].

Paired, v4c (round 4 final) minus shipped Stage 2 (large): recovered +58.5 [+53.1, +63.7], fixed +37.1 [+32.0, +42.3], sound +42.8 [+36.9, +48.5], rule violations -5.1 [-6.3, -3.9].

## medium omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shipped Stage 2 | 369 | 70% | 9% | 11% | 38.7 | 60% | 78% | 94% | 45% | 2.45 | 22.8 | 11.5 | 2.6 | 8.2 | 0.5 | 74% |
| v3c (round 2 final, accuracy first) | 369 | 88% | 71% | 54% | 43.6 | 75% | 95% | 52% | 37% | 1.03 | 17.0 | 8.2 | 1.4 | 7.1 | 0.4 | 86% |
| v4b (most sound on tuning) | 369 | 88% | 63% | 40% | 42.6 | 94% | 95% | 97% | 87% | 0.75 | 19.5 | 9.1 | 1.9 | 7.8 | 0.6 | 80% |
| v4c (round 4 final) | 369 | 88% | 63% | 40% | 42.9 | 91% | 96% | 96% | 85% | 0.82 | 19.8 | 9.5 | 1.8 | 7.8 | 0.6 | 79% |
| *parser's map given (nothing filled)* | 369 | - | - | - | - | 56% | 71% | 95% | 40% | 2.74 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 369 | - | - | - | - | 71% | 84% | 94% | 58% | 2.46 | 27.0 | 12.7 | 3.3 | 10.1 | 1.0 | - |
| *true layout* | 369 | - | - | - | - | 86% | 93% | 97% | 78% | 0.40 | 7.5 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| shipped Stage 2 | 6% | 39% | 55% |
| v3c (round 2 final, accuracy first) | 20% | 17% | 63% |
| v4b (most sound on tuning) | 58% | 29% | 13% |
| v4c (round 4 final) | 57% | 28% | 15% |

Paired, v4c (round 4 final) minus v3c (round 2 final, accuracy first) (medium): recovered -7.6 [-10.8, -4.6], fixed -13.3 [-17.1, -9.5], sound +48.0 [+42.8, +52.8], rule violations +2.7 [+2.0, +3.4].

Paired, v4c (round 4 final) minus v4b (most sound on tuning) (medium): recovered +0.3 [-2.2, +2.7], fixed +0.0 [-1.9, +1.9], sound -1.9 [-4.1, +0.0], rule violations +0.3 [+0.1, +0.5].

Paired, v4c (round 4 final) minus shipped Stage 2 (medium): recovered +53.7 [+48.2, +59.1], fixed +29.5 [+24.7, +34.4], sound +40.1 [+35.0, +45.5], rule violations -3.0 [-3.8, -2.2].

## intact drawings (nothing erased)

| version | n | drawings with a region flag | added components / drawing |
|---|---|---|---|
| shipped Stage 2 | 371 | 2.2% | 0.87 |
| v3c (round 2 final, accuracy first) | 371 | 6.7% | 1.25 |
| v4b (most sound on tuning) | 371 | 6.7% | 0.94 |
| v4c (round 4 final) | 371 | 6.7% | 0.95 |

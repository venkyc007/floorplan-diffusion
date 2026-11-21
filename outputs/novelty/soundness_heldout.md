# Round 4 held-out result: accuracy and structural soundness (151 held-out val drawings, kernel v16)

End to end, nothing given. Held-out drawings: never used to choose anything. v4c was chosen on the tuning drawings by the rule in docs/decisions.md (round 4) and frozen before this run; v3c and v4b run alongside as decided before the run. Each version ran once.

Soundness = the three structural checks of the review on the omission's region (src/novelty/soundness.py), judged without the true layout. Rule check = the project's 18-rule check (violation blobs in a window around the omission), in four groups; 'valid vs oracle' = no more violations than the parser's map with the true layout pasted into the erased area.

## large omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3c (round 2 final) | 151 | 99% | 80% | 55% | 44.3 | 57% | 91% | 49% | 24% | 1.74 | 22.3 | 10.8 | 2.1 | 9.1 | 0.3 | 92% |
| v4b (most sound on tuning) | 151 | 99% | 71% | 47% | 42.2 | 93% | 95% | 93% | 83% | 1.07 | 25.2 | 11.4 | 2.9 | 10.4 | 0.6 | 87% |
| v4c (round 4 final) | 151 | 99% | 74% | 44% | 42.8 | 87% | 95% | 94% | 81% | 1.15 | 24.8 | 11.7 | 2.4 | 10.1 | 0.6 | 90% |
| *parser's map given (nothing filled)* | 151 | - | - | - | - | 43% | 67% | 93% | 31% | 3.98 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 151 | - | - | - | - | 63% | 84% | 89% | 48% | 3.85 | 35.5 | 16.1 | 4.5 | 13.7 | 1.2 | - |
| *true layout* | 151 | - | - | - | - | 80% | 91% | 98% | 70% | 0.78 | 10.2 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| v3c (round 2 final) | 17% | 7% | 76% |
| v4b (most sound on tuning) | 65% | 19% | 17% |
| v4c (round 4 final) | 64% | 17% | 19% |

Paired, v4c (round 4 final) minus v3c (round 2 final) (large): recovered -6.6 [-11.9, -2.0], fixed -10.6 [-15.9, -6.0], sound +57.0 [+47.7, +65.6], rule violations +2.5 [+1.3, +3.7].

Paired, v4b (most sound on tuning) minus v3c (round 2 final) (large): recovered -9.3 [-15.2, -4.0], fixed -7.9 [-13.2, -3.3], sound +59.6 [+51.0, +67.5], rule violations +2.9 [+1.7, +4.1].

Paired, v4c (round 4 final) minus v4b (most sound on tuning) (large): recovered +2.6 [-2.6, +7.9], fixed -2.6 [-5.3, -0.7], sound -2.6 [-6.6, +1.3], rule violations -0.5 [-0.9, -0.0].

## medium omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v3c (round 2 final) | 151 | 95% | 79% | 57% | 46.9 | 75% | 91% | 60% | 42% | 0.79 | 14.5 | 7.2 | 1.3 | 5.8 | 0.3 | 87% |
| v4b (most sound on tuning) | 151 | 95% | 72% | 52% | 44.1 | 91% | 95% | 99% | 87% | 0.49 | 17.2 | 8.2 | 1.8 | 6.8 | 0.4 | 78% |
| v4c (round 4 final) | 151 | 95% | 71% | 51% | 44.5 | 87% | 95% | 99% | 83% | 0.57 | 17.2 | 8.6 | 1.5 | 6.7 | 0.4 | 80% |
| *parser's map given (nothing filled)* | 151 | - | - | - | - | 50% | 71% | 99% | 41% | 2.42 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 151 | - | - | - | - | 72% | 85% | 99% | 63% | 2.03 | 23.6 | 11.1 | 2.9 | 8.9 | 0.7 | - |
| *true layout* | 151 | - | - | - | - | 87% | 97% | 98% | 83% | 0.46 | 6.5 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| v3c (round 2 final) | 31% | 11% | 58% |
| v4b (most sound on tuning) | 68% | 20% | 13% |
| v4c (round 4 final) | 62% | 21% | 17% |

Paired, v4c (round 4 final) minus v3c (round 2 final) (medium): recovered -8.6 [-13.9, -4.0], fixed -6.0 [-11.3, -0.7], sound +41.7 [+33.1, +50.3], rule violations +2.7 [+1.8, +3.8].

Paired, v4b (most sound on tuning) minus v3c (round 2 final) (medium): recovered -7.3 [-12.6, -2.6], fixed -4.6 [-9.9, +0.7], sound +45.7 [+37.1, +54.3], rule violations +2.7 [+1.7, +3.8].

Paired, v4c (round 4 final) minus v4b (most sound on tuning) (medium): recovered -1.3 [-5.3, +2.0], fixed -1.3 [-4.6, +2.0], sound -4.0 [-7.9, -0.7], rule violations +0.0 [-0.3, +0.4].

## intact drawings (nothing erased)

| version | n | drawings with a region flag | added components / drawing |
|---|---|---|---|
| v3c (round 2 final) | 151 | 3.3% | 0.93 |
| v4b (most sound on tuning) | 151 | 3.3% | 0.73 |
| v4c (round 4 final) | 151 | 3.3% | 0.72 |

# Round 4 TEST result: v4c on the CubiCasa5K test drawings (one run, approved by the user 2026-11-21; kernel v17)

End to end, nothing given. Every typed test drawing (parser_test.zip) except the check set; one small, one medium and one large omission per drawing, plus the intact drawing. v4c frozen on 2026-11-21 before this run (chosen on tuning val, checked on held-out val); nothing changed after it. Only v4c was run on test.

Soundness = the three structural checks of the review on the omission's region (src/novelty/soundness.py), judged without the true layout. Rule check = the project's 18-rule check (violation blobs in a window around the omission), in four groups; 'valid vs oracle' = no more violations than the parser's map with the true layout pasted into the erased area.

## large omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4c (round 4 final) | 369 | 98% | 66% | 47% | 40.6 | 86% | 96% | 92% | 78% | 1.38 | 27.6 | 12.9 | 2.9 | 11.1 | 0.7 | 87% |
| *parser's map given (nothing filled)* | 369 | - | - | - | - | 48% | 66% | 90% | 31% | 4.39 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 369 | - | - | - | - | 59% | 78% | 89% | 44% | 4.12 | 39.9 | 18.2 | 5.4 | 15.0 | 1.3 | - |
| *true layout* | 369 | - | - | - | - | 77% | 93% | 96% | 68% | 0.71 | 11.3 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| v4c (round 4 final) | 54% | 24% | 22% |

## medium omissions

| version | n | found | recovered (2 px) | fixed (strict) | tol. mIoU | walls connect | openings in walls | rooms closed | sound | mixed rooms | rule violations | walls (C1-C5) | openings (C9-C13) | rooms (C6-C8, R1, R3) | fixtures / railings (C14-C18) | valid vs oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v4c (round 4 final) | 369 | 88% | 63% | 40% | 42.9 | 91% | 96% | 96% | 85% | 0.82 | 19.8 | 9.5 | 1.8 | 7.8 | 0.6 | 79% |
| *parser's map given (nothing filled)* | 369 | - | - | - | - | 56% | 71% | 95% | 40% | 2.74 | - | - | - | - | - | - |
| *oracle (true layout pasted into the erased area)* | 369 | - | - | - | - | 71% | 84% | 94% | 58% | 2.46 | 27.0 | 12.7 | 3.3 | 10.1 | 1.0 | - |
| *true layout* | 369 | - | - | - | - | 86% | 93% | 97% | 78% | 0.40 | 7.5 | - | - | - | - | - |

Sound fills split by whether they match the true layout (recovered within 2 px): a sound fill that does not match is a different but structurally valid layout (the review's 'plausible alternative').

| version | sound and matching the truth | sound, different layout | not sound |
|---|---|---|---|
| v4c (round 4 final) | 57% | 28% | 15% |

## intact drawings (nothing erased)

| version | n | drawings with a region flag | added components / drawing |
|---|---|---|---|
| v4c (round 4 final) | 371 | 6.7% | 0.95 |

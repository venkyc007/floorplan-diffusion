## How it works

**Stage 1 — you say where, the model redraws it.** A plan is a 128 × 128 map with 6 classes. A U-Net draws the
missing region in one pass (the *sketch*); the sketch is then noised back to a level t₀ (400 / 600 / 700 for small /
medium / large holes) and a diffusion model, fine-tuned with structural losses (wall connectivity, room closure),
denoises it while the known part of the plan is held fixed. The U-Net is accurate; the diffusion finish makes the
structure plausible (walls connect, rooms close).

**Stage 2 — the system has to find what is missing.**

1. **PARSE** — a segmentation network (U-Net with a ResNet-34 encoder, trained on CubiCasa5K drawings) turns the
   drawing into a 16-class map: wall, door, window, railing, six room types, outdoor, background, four fixture types.
2. **CHECK** — architectural rules and two learned detectors look for what a complete plan would have: a room with no
   door, an open gap where a door belongs, a window symbol that is not drawn, a long exterior wall with no window, a
   room holding both a toilet and a stove. Each flag has a type and a location.
3. **FILL** — around each flag, a 16-class diffusion model (the same hybrid as Stage 1, warm-started from it) redraws
   the region; only the flagged type is kept. Large regions use models fine-tuned on large holes.
4. **VERIFY** — several samples are drawn and the one with the fewest structural rule violations wins.

If CHECK misses something, mark it yourself (2b): then only PARSE and FILL run.

## How well it works

| | result |
|---|---|
| Stage 1, test set (7,774 cases) | tolerant mIoU 0.846, structurally valid 88.0% (U-Net alone 0.850 / 81.0%) |
| Stage 2 PARSE, held-out val drawings | 16-class mIoU 70.1; door / window F1 79.5 / 77.3 |
| Stage 2 FILL, hole given, test (120 holes) | the diffusion finish makes 52% of fills structurally valid (U-Net sketch alone 18%) |
| Stage 2 CHECK, test (180 drawings) | 70% of omissions found, 0.80 false alarms per complete drawing |
| Stage 2 end to end, test (120 drawings, a third whole rooms / regions) | 66% found, 12% fixed, 0.85 false additions per drawing |
| Stage 2 end to end, single components only, test (all 371 drawings) | 62% found, 41% fixed (22% by the fill itself), 0.92 false additions per drawing |

Honest limits: the automatic mode rarely completes an unseen drawing correctly, so for a known gap use manual mode
(mark the region); room types rely partly on the printed room names; false alarms are higher on new drawings. Full numbers and sources:
`docs/results.md` in the repository.

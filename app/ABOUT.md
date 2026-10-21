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
| Stage 2 CHECK, held-out val drawings | 60% of omissions found, 0.59 false alarms per complete drawing |
| Stage 2 FILL, hole given (val) | small holes 25/30 fixed, large holes 29/30 (B4) |
| Stage 2 end to end, 12 fresh val drawings | 9 found, 8 fixed, 5 false additions |

Honest limits: a whole missing room is not found automatically (use manual mode); room types rely partly on the
printed room names; false alarms are higher on new drawings than on the tuning drawings. Full numbers and sources:
`docs/results.md` in the repository.

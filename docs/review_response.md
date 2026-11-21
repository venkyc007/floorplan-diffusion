# Review feedback (novelty assessment): point by point

Material for the report / slides (the report is the user's own writing; this file gives the facts, numbers and
where they come from). Review = instructor's comments + TA's comments after the preparatory review. Non-coding
points (poster, slide pacing) are handled separately and are only listed.

Status key: DONE = evidence exists in the repo. All coding points are DONE (2026-11-21); held-out numbers below.

| # | review point | what answers it | evidence | status |
|---|---|---|---|---|
| 1 | Handle large omissions (main priority) | Novelty rounds 1-2: structure net + large-hole fill, end to end (no mask). Round 4 (final v4c): the fills are made structurally sound. Held-out large omissions: recovered 11% (shipped) -> 80% (v3c) -> 74% (v4c), strict fixed 13% -> 55% -> 44%, sound 24% (v3c) -> 81% (v4c; true layout 70%) | `docs/novelty.md` sections 5, 8, 10; `outputs/novelty/e2e_heldout_round2.md`; `outputs/novelty/soundness_heldout.md` | DONE |
| 2 | Explain how the constraints affect reconstruction (main priority) | Ablation: the same pipeline with each use of the constraints switched on in turn (U-Net sketch -> constraint-trained diffusion -> verifier -> clean-up -> repair -> selection by soundness), all measured by accuracy, the three checks and the rule check by group; plus one plan pictured | `outputs/novelty/soundness_tuning.md` (kernels v14, v15; `docs/novelty.md` 10.3), `outputs/novelty/assessment/constraints_*.png`; Stage 2 test: U-Net alone vs + constrained diffusion vs control diffusion (`outputs/stage2/final_test.md`). Finding: the constraint-trained diffusion cuts rule violations 43 -> 25 per case but not the structural problems (sound 27% -> 25%); the rule-based repair by the three checks makes fills sound (25% -> 75%), selection by soundness adds +5 | DONE |
| 3 | Break the 46% completion rate down by omission size | The 46% = second Stage 2 test evaluation (single components, railings left out); split by area, plus the size-balanced test and the large-omission results | `outputs/novelty/completion_by_size.md` (`src/novelty/completion_by_size.py`) | DONE |
| 4 | Define what counts as a completed omission | found + right class back at the spot with IoU >= 0.3 (strict); for walls also "recovered within 2 px"; new: "sound" | `outputs/novelty/completion_by_size.md` top section | DONE |
| 5 | For large omissions show corrupted plan, reconstruction and violation map together | Figure per case: corrupted drawing, reconstruction (drawn and as a map), violation map of the reconstruction, of the corrupted plan and of the true layout, with the three checks PASS / FAIL; first K held-out cases, no picking | `outputs/novelty/assessment/large_heldout_*.png` (`src/novelty/assessment_figures.py large`; v4c, first 6 held-out cases, 5 of 6 sound) | DONE |
| 6 | Show whether a plausible alternative preserves wall connectivity, opening placement and room closure (one ground truth captures only part of the task) | The three checks are implemented exactly as named and judge a plan WITHOUT the true layout; reported for the fill, the corrupted plan, the oracle and the true layout; "sound but not matching the truth" counted; several diffusion samples of one omission shown, each with its checks | `src/novelty/soundness.py`, `outputs/novelty/soundness_*.md`, `outputs/novelty/assessment/alternatives_*.png`. Held-out, v4c, large: 64% sound and matching the truth, 17% sound with a different layout, 19% not sound; walls connect 87%, openings in walls 95%, rooms closed 94% | DONE |
| 7 | Explain why the final eight constraints were chosen; one floor plan to illustrate their effect | The eight = the parser v2 fine-tune losses (C3, C6, C9, C10, C11, C12, C15, C16); why (below); the fill uses 14, the verifier 17 + R1 + R3 | this file, section "The eight constraints"; `docs/decisions.md` (gates G2, G3; round 4); `outputs/stage2/constraint_weights.md` | DONE (reason confirmed by the user, 2026-11-21) |
| 8 | Report the experimental settings / modest parsing performance | Parser v2B on 159 held-out drawings: mIoU 70.1, wall IoU 77.6 | `outputs/stage2/parser_v2.md`, `docs/results.md` section 3 | DONE (existing) |
| 9 | Brisker walkthrough of the early pipeline; more time for results | presentation | - | user (non-coding) |

## Finding that changes how validity is measured

The project's "valid" (no more rule violations than the clean TRUE drawing) cannot judge a fill on a real drawing:
the parser's map around the hole carries its own violations. Pasting the true layout into the erased area of the
parser's map (the ORACLE, a perfect fill) scores 29.7-31.0 violations per large case on 30-40 tuning cases, more
than the fills (20-25); the clean true drawing scores about 9. So "valid" was near 0% even for a perfect fill.
Tables now give "valid vs oracle" and, above all, the three structural checks of point 6, which look only at the
elements in the omission's region and need no ground truth. The true layout passes them in about 85% of tuning
cases (real drawings have free-standing walls, doors left out, glazed inner walls).

## The eight constraints

Stage 2 has 18 constraints (C1-C18, `docs/constraints_spec.md`), each a differentiable violation map. They are used
in four places, with different subsets:

| where | constraints | why this subset |
|---|---|---|
| parser v2 fine-tune (the "eight") | C3 axis-aligned walls, C6 different spaces never touch, C9 openings in a wall, C10 openings cut through the wall, C11 a door joins two spaces, C12 a window has inside / outside, C15 fixture in a compatible room, C16 fixture against a wall | the user's choice (2026-10-19, "local constraint losses"). Reading of the code: these work on a whole drawing with no hole and no size in pixels. Left out: rules defined around a hole (C2 floating walls, C5 walls continue into the hole, C8 reachability, C17 required fixtures, C18 railing) - the parser sees no hole; rules that measure a size (C4 thickness, C7 width, C13 opening length, C14 door clearance) - the parser works at ~8 px walls, the rules at 3 px; C1 (skeleton end points, GT-relative) |
| fill diffusion training | 14: all but C2, C5, C10, C13 | gate G3: all 18 at full strength cost 6.1 points of accuracy; those four hit the weight cap in the gradient-size calibration; without them the loss passed (-1.8 points, within seed spread) |
| verifier (best-of-N) and the rule check | 17 (all but C17, which needs the ground truth) + R1 room area + R3 forbidden door pairs | everything that can be computed on a window |
| detection (CHECK) | violation maps as detector inputs; rules C8, R5, R6, R8, ink checks | chosen on tuning drawings for a false-alarm budget |

Gate G2 (1,000 real windows) decided each rule's mode: where more than 10% of real windows already break a rule
(e.g. C4 thickness 43%, C7 width 41%, C12 windows 24%), the rule only penalises being WORSE than the real plan.

Confirmed by the user (2026-11-21): the reading above is right, and a second reason applies across the project -
rules were dropped where they added nothing because the learned model already obeyed them. The records show this:
in the gradient-size calibration (`outputs/stage2/constraint_weights.md`, 50 batches of real windows) four rules
gave almost no training signal - C10 (median gradient 8.3e-6), C13 (1.0e-4), C5 (2.7e-4), C2 (5.5e-4), against
1.7e-3 for the reconstruction loss - i.e. the model rarely broke them. Forcing them in (their weights hit the cap)
cost accuracy: all 18 rules at full strength -6.1 points (gate G3), the 14 without them -1.8 points (within seed
spread). C2, C5 and C13 are also outside the parser's eight; C10 is in the parser's set (it is measured at the
parser's own scale there). Related evidence: in the Stage 2 test, switching the constraint losses off in the fill
changed nothing measurable (`outputs/stage2/final_test.md`, "Ablation: constrained vs control diffusion"), and in round 4 the
constraint-trained diffusion lowered the rule count (43 -> 25 per case) but not the structural problems
(`docs/novelty.md` 10.3).

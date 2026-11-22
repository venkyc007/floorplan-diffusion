# Novelty assessment: poster + 2-page report — where the material is

Course instructions: `IE643 2026 Course Project Novelty Assessment Instructions.pdf` (the student's Downloads folder,
not in the repo). Deadline for upload (poster PDF + report PDF to the shared Google Drive folder): **22 Nov 2026,
23:59 IST**. Poster printed on **A1** by the team; presented in person on 23 or 25 Nov (10:00–13:00). File names:
`One Hot Mencoding_IE643_CourseProject_Poster.pdf` and `One Hot Mencoding_IE643_NoveltyAssessment_report.pdf`
(the PDF's pattern is `Teamname_IE643_...`; check whether the team name should keep its spaces).

**Integrity rule (IE643):** the report and the poster text must be the student's own writing; copied or
AI-generated text scores zero (`docs/report_guide.md`). This guide and the repo provide material, figures, numbers
and checks. Any assistant helping with the poster should do layout, figures, tables and checks of numbers, and
leave the sentences to the student.

## Still needed from the student (not in the repo)

| item | why |
|---|---|
| Name, roll number, IITB e-mail (poster header lines 4–5) | header |
| The course poster template (the PDF's "link"; Google Slides / PPT) with the IITB and IE 643 logos | header + layout; the PDF says to use it |
| GitHub link for the QR code: https://github.com/venkyc007/floorplan-diffusion — the novelty work is on branch `novelty` (not merged into `main`), so either merge first or link https://github.com/venkyc007/floorplan-diffusion/tree/novelty | item 12 |
| Demo video link for the QR code (existing Stage 2 video, or a new one showing the app's novelty tab) | item 13 |
| Acknowledgements | item 11 |
| QR codes (no QR package in the project env; any QR generator works) | items 12–13 |

## Poster sections -> material

| # | section | material |
|---|---|---|
| 1 | Header | above |
| 2 | Abstract | task: find + redraw large missing parts of real floor-plan drawings, nothing given; headline test numbers below |
| 3 | Introduction | `docs/report_guide.md` (Introduction row); why large omissions: `docs/novelty.md` section 1 (the shipped system finds 80% of them but redraws almost none) |
| 4 | Workflow flowchart | `docs/report_figures/fig01_workflow.png` (Stage 1 -> Stage 2), then **`outputs/novelty/poster/novelty_workflow.png`** (rounds 1–4 with held-out numbers) |
| 5 | Methods (full combined architecture) | **`outputs/novelty/poster/hybrid_pipeline.png`**; Stage 2 parts `docs/report_figures/fig06_stage2_pipeline.png`; details `docs/novelty.md` 2 (round 1), 8.2 (structure net, ink / stroke maps, rays), 10.2 (clean-up, repair, selection by soundness) |
| 6 | Dataset table | `outputs/stage2/data_stats.md` (CubiCasa5K: 5,000 plans, official split 4,200 / 400 / 400; 16 classes, `configs/stage2/classes16.yaml`); novelty training set `docs/novelty.md` 8.2 ("Bigger set": 3,944 train drawings, 35,339 erased + 3,944 intact versions); evaluation: 212 tuning / 151 held-out val drawings, 369 test drawings per size; omission sizes small < 2.7%, medium 2.7–7.7%, large >= 7.7% of the building |
| 7 | Novelty assessment (tasks attempted) | `docs/novelty.md` sections 1–10; review points and what was done: `docs/review_response.md`; constraint ablation (how each use of the rules changes the result): `docs/novelty.md` 10.3 + `outputs/novelty/assessment/constraints_tuning_*.png` |
| 8 | Results | quantitative: **`outputs/novelty/poster/results_heldout.png`**, **`results_test.png`** (all four versions on test), `outputs/novelty/soundness_test.md`, `soundness_heldout.md`, `completion_by_size.md`; qualitative: **`outputs/novelty/poster/qualitative_successes.png`** (three full successes of v4c, chosen by a fixed rule; the caption gives the success rate: 37% held-out, 30% test; do not present them as typical), `outputs/novelty/assessment/large_heldout_*.png` (corrupted / reconstruction / problem maps), `alternatives_tuning_*.png` (sound but different layouts), app screenshots of the novelty tab; Stage 1 / Stage 2 (intensive assessment): `docs/results.md`, `docs/report_figures/` |
| 9 | Conclusions + future work | `docs/novelty.md` 8.6, 10.5–10.6 (honest readings); not attempted: room-graph idea (`docs/novelty.md` 6 / report 8.2); generalisation gap held-out -> test |
| 10 | References | `docs/report_guide.md` "References" (already in the course's format; add LaMa for the Fourier blocks if they are mentioned: Roman Suvorov et al. Resolution-robust Large Mask Inpainting with Fourier Convolutions, WACV, 2022. https://arxiv.org/abs/2109.07161) |
| 11–13 | Acknowledgements, GitHub QR, demo QR | student |

## Headline numbers (all from saved per-case results; brackets = Wilson 95% intervals)

Large omissions (>= 7.7% of the building), end to end, nothing given. Final hybrid = **v4c**.

| | held-out val (151) | TEST (369, one run) |
|---|---|---|
| found | 99% | 98% |
| recovered (2 px) | 74% [66, 80] | 66% [61, 71] |
| fixed (strict IoU >= 0.3) | 44% [37, 52] | 47% [42, 53] |
| sound (3 structural checks) | 81% [74, 86] | 78% [74, 82] |
| true layout's own soundness | 70% | 68% |
| intact drawings with a false region flag | 3.3% | 6.7% [5, 10] |

All versions on TEST, large omissions (369, same cases, each run once; `outputs/novelty/soundness_test.md`):

| TEST, large | found | recovered (2 px) | fixed (strict) | sound |
|---|---|---|---|---|
| shipped Stage 2 (baseline) | 82% | 8% | 10% | 36% |
| v3c (accuracy first) | 98% | 78% | 56% | 19% |
| v4b (most sound) | 98% | 68% | 49% | 85% |
| **v4c (final)** | 98% | 66% | 47% | 78% |
| true layout | - | - | - | 68% |

Medium (369): shipped 9 / 11 / 45, v3c 71 / 54 / 37, v4b 63 / 40 / 87, v4c 63 / 40 / 85 (recovered / fixed /
sound). Small: fixed 40-41% for all. Intact flags: shipped 2.2%, hybrids 6.7%. Points to state honestly (the
student's words): v4b came out more sound than v4c on test at the same accuracy, but v4c was chosen on tuning before
any test run and is kept; v3c keeps its held-out accuracy on test, v4c loses 8 points of recovered (the round-4
rules, not the models). `docs/novelty.md` 10.6. Metric definitions: `outputs/novelty/completion_by_size.md` (top).

## 2-page report ("approaches tried and experiments for novelty")

Material in order: problem (novelty.md 1) -> round 1 locate + fill (2, 4–5) -> round 2 structure first (8) ->
round 3 validity clean-up (9, a negative result: rule check not better) -> round 4 soundness checks, repair,
selection; the rule written before the results; v4b failed its guard, v4c chosen (10.1–10.4) -> held-out and test
(10.5–10.6) -> limits (ink-map caveat, novelty.md 8.5: the numbers are for "a part of the plan not drawn at all";
generalisation gap). Protocol: everything chosen on 212 tuning drawings, each version frozen and run once on 151
held-out drawings, then once on test.

## The demo app (for the video / the presentation)

`python demo_app/app.py` -> tab "Novelty · large omissions": held-out example cases (with what each version did on
them in the evaluation), version choice v4c / v4b / v3c, "Compare all three", structural problems drawn as rings.
Needs the novelty checkpoints (see `demo_app/README.md`, "Novelty tab").

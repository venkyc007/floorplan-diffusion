# Report and slides — where the material is

The IE643 rules: the report must use the course template, be the team's own writing (plagiarism check; zero for
copied or AI-generated text), include the GitHub link, distinguish work **before** and **after** the prep-presentation
review, and cite references in the given format. This guide only points to material in the repo — the text is yours.

**Before the prep review = Stage 1** (ResPlan, mask given). **After the prep review = Stage 2** (the TA's clarified
scope: real drawings, find what is missing, 10–15 component types, "every room must have a door").

## Report sections → material

| section | material |
|---|---|
| Abstract | `README.md` table (task, method, headline numbers); Stage 1 test numbers; Stage 2 numbers (val, then test after the run) |
| Introduction | why incomplete plans matter (drafting errors, damaged scans, plan digitisation); why diffusion (multiple plausible completions, inpainting) + constraints (architectural validity) |
| Workflow | `docs/decisions.md` section headings in date order (Day 1 data → Day 8 app → Stage 2 data, constraints, parser, CHECK, FILL, pipeline, final test); a diagram: data → corruption → models → selection → app; then PARSE → CHECK → FILL → VERIFY |
| Related work | see the reference list below; at least 2–3 papers described in your own words (what they do, what you took from them) |
| Methods — before the review | `docs/stage1_summary.md`; models `src/stage1/models/` (U-Net CE, Palette-style diffusion, clDice / closure losses); the hybrid and size-adaptive t₀; the lesson on constraint weights (100× too strong by loss value → calibrated by gradient size) |
| Methods — after the review | `docs/results.md` §1; `docs/constraints_spec.md` (C1–C18, R1–R4); parser `src/stage2/parse/`; CHECK `src/stage2/detect/` (rules + detectors v2 / v3, tuning budget); FILL `src/stage2/complete/` (warm start 6 → 16 classes, hybrid sampler, B4 large holes); VERIFY (best-of-N) |
| Data | Stage 1: `docs/data_notes.md` (ResPlan, exclusion of augmented plans, dedup, split sizes, corruption types, benchmarks); Stage 2: `outputs/stage2/data_stats.md`, `data_overlay.png`, `configs/stage2/classes16.yaml` (16 classes, mapping), official split + tuning / held-out ids, erasure (`erasure_check.png`, `erasure_large_examples.png`) |
| Experiments | training settings in `docs/code_walkthrough.md` part 9 (optimizer, lr, warm-up + cosine, steps, batch); hardware: Kaggle T4 × 2 (16 GB each), laptop Windows 16 GB RAM, no GPU; all ablations: constrained vs control (Stage 1 `outputs/stage1/finetune_ablation.md`; Stage 2 in the final test), U-Net vs hybrid (`t0_sweep.md`), guidance (`eta_tune.md`), B4 (`b4_val.md`), CHECK parts (`check_tune.md`); GitHub link |
| Results | `docs/results.md` (tables per step, with the data role of each number); figures below; Stage 1 test `outputs/stage1/stage1_test.md`; Stage 2 test `outputs/stage2/final_test.md` (after the run); qualitative: e2e figures, `outputs/stage1/interface/hybrid_samples*.png` (local, not committed) |
| Plan for novelty assessment | ideas: fix the scale mismatch consistently and retrain the detectors; real (not synthetic) omission data; vector / graph output with topological constraints; a learned verifier; room typing without printed text |
| Conclusion | what worked (Stage 1 structure gains; parser; small-component completion), what did not (large omissions, false alarms on new drawings, constraint losses with small effect) — `docs/results.md` §4 |
| References | below |

## Figures worth using (paths in the repo)

| figure | shows |
|---|---|
| `outputs/stage1/corruption_examples.png` | Stage 1 corruption types |
| `outputs/stage1/pareto.png`, `refine_tradeoff.png` | Stage 1 selection: accuracy vs structure |
| `outputs/stage1/arch_violations.png` | the rule violations the diffusion removes |
| `outputs/stage1/interface/hybrid_samples.png` | Stage 1 qualitative, small / medium / large (local file, commit only if you want it public) |
| `outputs/stage2/data_overlay.png` | drawing → labels at two scales, 16-class legend |
| `outputs/stage2/constraint_toys.png` | every constraint's violation map on a toy plan |
| `outputs/stage2/erasure_check.png`, `erasure_large_examples.png` | how omissions are simulated |
| `outputs/stage2/e2e_val_batch2_b4.png` (crop 3–4 rows) | end to end: drawing → parse + flags → result → truth |
| `outputs/stage2/check_set_holes_v4.png` | close-ups: parse / raw fill / after the type restriction |
| `outputs/stage2/large_fill_compare_v2.png` | large holes: successes and failures |
| `outputs/stage2/sketch_prior_examples.png` | the sketch prior |
| app screenshots (Stage 2 tab) | the four step pictures |

## Slides (max 18 + references) — a possible plan

1 title (task, "IE 643 course project", team name **One Hot Mencoding**, names, roll numbers) · 2 outline · 3 problem statement ·
4 workflow diagram · 5–6 before the prep review (Stage 1 method + test results) · 7 after the prep review (what
changed and why) · 8–10 proposed approach (PARSE, CHECK, FILL + VERIFY) · 11 data and preprocessing ·
12 experimental setup · 13–16 results (Stage 1 quantitative, Stage 2 per step, end to end, qualitative) ·
17 conclusions · 18 plan for novelty assessment · references.

## References (check each against the paper's own page before submitting)

- Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising Diffusion Probabilistic Models, NeurIPS, 2020. https://arxiv.org/abs/2006.11239
- Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models, ICLR, 2021. https://arxiv.org/abs/2010.02502
- Alex Nichol and Prafulla Dhariwal. Improved Denoising Diffusion Probabilistic Models, ICML, 2021. https://arxiv.org/abs/2102.09672 (cosine schedule)
- Chitwan Saharia, William Chan, Huiwen Chang, Chris A. Lee, Jonathan Ho, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Palette: Image-to-Image Diffusion Models, SIGGRAPH, 2022. https://arxiv.org/abs/2111.05826
- Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. RePaint: Inpainting using Denoising Diffusion Probabilistic Models, CVPR, 2022. https://arxiv.org/abs/2201.09865
- Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI, 2015. https://arxiv.org/abs/1505.04597
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition, CVPR, 2016. https://arxiv.org/abs/1512.03385
- Suprosanna Shit, Johannes C. Paetzold, Anjany Sekuboyina, Ivan Ezhov, Alexander Unger, Andrey Zhylka, Josien P. W. Pluim, Ulrich Bauer, and Bjoern H. Menze. clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation, CVPR, 2021. https://arxiv.org/abs/2003.07311
- Ahti Kalervo, Juha Ylioinas, Markus Häikiö, Antti Karhu, and Juho Kannala. CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis, SCIA, 2019. https://arxiv.org/abs/1904.01920
- Nelson Nauata, Kai-Hung Chang, Chin-Yi Cheng, Greg Mori, and Yasutaka Furukawa. House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation, ECCV, 2020. https://arxiv.org/abs/2003.06988
- Mohammad Amin Shabani, Sepidehsadat Hosseini, and Yasutaka Furukawa. HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising, CVPR, 2023. https://arxiv.org/abs/2211.13287
- ResPlan dataset (authors as listed in the repository). https://github.com/m-agour/ResPlan, Accessed on: <date>.
- CubiCasa5K Kaggle mirror. https://www.kaggle.com/datasets/qmarva/cubicasa5k, Accessed on: <date>.
- Patrick von Platen et al. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/diffusers, Accessed on: <date>.
- Pavel Iakubovskii. Segmentation Models PyTorch. https://github.com/qubvel/segmentation_models.pytorch, Accessed on: <date>.
- Project code: https://github.com/venkyc007/floorplan-diffusion (must be public or shared with the instructor and TAs).

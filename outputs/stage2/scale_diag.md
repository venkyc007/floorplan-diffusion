# Scale mismatch: what it costs the FILL (hole given, val)

30 small + 30 large holes on completion_val plans (seed 91). The SAME plan and hole filled at the training scale and at the pipeline's scale (plan resized by R = 0.838, nearest neighbour, as run_pipeline's maps are). Ground-truth labels: only the scale changes. Fast settings. Script: src/stage2/complete/scale_diag.py.

How to read it: A compares each fill with the truth at its own scale. Its tolerant mIoU leans towards the pipeline scale (the 1 px tolerance is relatively larger when walls are thinner), so 'fixed' (main-class component IoU >= 0.3) and validity are the fairer columns there. B scores both at the training scale; it includes the loss from resizing itself (see the ceiling row), which leans towards the training scale. A real scale fix would avoid that resizing loss (FILL would get its own map at the training scale), so A's fixed / valid columns are the best guide to what a fix could gain.

## A. Each fill against the truth at its own scale (the model's quality at each scale)

| holes | arm | n | tol. mIoU training scale | tol. mIoU pipeline scale | difference [95% CI] | fixed training / pipeline | valid training / pipeline |
|---|---|---|---|---|---|---|---|
| small | U-Net sketch | 30 | 75.9 | 76.2 | +0.3 [-3.4, +4.2] | 80% / 77% | 20% / 20% |
| small | final | 30 | 77.8 | 75.8 | -2.0 [-5.9, +1.7] | 80% / 80% | 77% / 73% |
| large | final | 30 | 46.6 | 46.7 | +0.1 [-2.3, +2.6] | 93% / 87% | 33% / 33% |

## B. Both scored at the training scale (the pipeline-scale fill resized back to it)

Includes the loss from resizing itself; the ceiling row is the truth resized by R and back with no model.

| holes | arm | n | tol. mIoU training scale | tol. mIoU pipeline scale | difference [95% CI] | fixed training / pipeline |
|---|---|---|---|---|---|---|
| small | U-Net sketch | 30 | 75.9 | 61.1 | -14.8 [-19.0, -10.1] | 80% / 73% |
| small | final | 30 | 77.8 | 60.1 | -17.8 [-21.6, -13.8] | 80% / 73% |
| small | resampling ceiling (truth resized and back) | 30 | 100.0 | 76.1 | -23.9 | 100% / 93% |
| large | final | 30 | 46.6 | 36.6 | -10.0 [-12.7, -7.6] | 93% / 63% |
| large | resampling ceiling (truth resized and back) | 30 | 100.0 | 74.1 | -25.9 | 100% / 97% |

By hole kind (final arm, own-scale tol. mIoU training -> pipeline scale):

| holes | kind | n | training | pipeline |
|---|---|---|---|---|
| large | region_large | 14 | 43.0 | 41.9 |
| large | room_whole | 16 | 49.8 | 51.0 |
| small | door | 7 | 93.9 | 92.5 |
| small | fixture | 6 | 63.1 | 57.0 |
| small | railing | 3 | 61.7 | 58.1 |
| small | region | 3 | 81.8 | 77.5 |
| small | room | 4 | 58.0 | 60.5 |
| small | wall | 4 | 86.7 | 84.1 |
| small | window | 3 | 96.7 | 100.0 |

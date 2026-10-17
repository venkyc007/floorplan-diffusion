# Gate G4: step time with vs without constraints

Full-size 16-class diffusion ([64, 128, 256, 256]), batch 32 windows of 128 x 128, median over 10 steps (forward + backward + AdamW, fp16 autocast on GPU), device: Tesla T4.

| variant | s / step |
|---|---|
| constraints off | 0.90 |
| constraints on | 1.40 |

Ratio on / off = 1.55 -> **FAIL** (limit 1.5). Flood-fill steps K = 96.

Steps the constrained run finishes in 10 h at this speed (minus ~5% for validation): 24399.

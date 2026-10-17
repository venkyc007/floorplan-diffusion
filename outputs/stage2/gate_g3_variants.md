# Gate G3 re-run: constraint variants (user, 2026-10-17)

Same tiny-model setup as G3 (64 fixed real windows, 400 steps, Adam 3e-3, batch 16, ramp 50). Variants on seed 0; OFF on seeds 0 and 1 for the chance spread. C/D drop the 4 capped constraints (C2, C5, C10, C13) from training and keep the others' calibrated weights. Pass: within max(2, OFF spread) points of OFF seed 0 on hole accuracy (2.0) and mIoU (2.0).

| run | hole acc | hole mIoU | d acc | d mIoU | pass |
|---|---|---|---|---|---|
| OFF_seed0 | 53.6 | 25.6 | +0.0 | +0.0 | - |
| OFF_seed1 | 52.3 | 24.3 | +0.0 | +0.0 | - |
| C_clean_rho1 | 51.8 | 24.5 | -1.8 | -1.1 | yes |
| D_clean_rho_half | 54.0 | 24.6 | +0.4 | -1.1 | yes |
| A_all_rho_half | 52.5 | 24.8 | -1.1 | -0.8 | yes |
| B_all_rho_quarter | 56.0 | 26.5 | +2.5 | +0.9 | yes |

OFF seed-to-seed spread: accuracy 1.3, mIoU 1.4 points.
**Chosen (first passing in order C, D, A, B): C_clean_rho1**

Mean gradient share per constraint (|| grad w_i L_i || / || grad total ||, at steps 100-400):

| constraint | C_clean_rho1 | D_clean_rho_half | A_all_rho_half | B_all_rho_quarter |
|---|---|---|---|---|
| C1 | 0.0027 | 0.0012 | 0.0011 | 0.0006 |
| C2 | - | - | 0.0047 | 0.0024 |
| C3 | 0.0002 | 0.0001 | 0.0001 | 0.0001 |
| C4 | 0.0016 | 0.0006 | 0.0006 | 0.0003 |
| C5 | - | - | 0.0157 | 0.0074 |
| C6 | 0.0196 | 0.0079 | 0.0077 | 0.0041 |
| C7 | 0.0022 | 0.0009 | 0.0008 | 0.0005 |
| C8 | 0.0101 | 0.0035 | 0.0038 | 0.0018 |
| C9 | 0.0013 | 0.0005 | 0.0006 | 0.0003 |
| C10 | - | - | 0.0132 | 0.0067 |
| C11 | 0.0067 | 0.0024 | 0.0026 | 0.0013 |
| C12 | 0.0189 | 0.0085 | 0.0109 | 0.0043 |
| C13 | - | - | 0.0068 | 0.0035 |
| C14 | 0.0034 | 0.0012 | 0.0012 | 0.0006 |
| C15 | 0.0014 | 0.0005 | 0.0005 | 0.0003 |
| C16 | 0.0005 | 0.0002 | 0.0002 | 0.0001 |
| C17 | 0.0125 | 0.0041 | 0.0057 | 0.0024 |
| C18 | 0.0021 | 0.0009 | 0.0008 | 0.0005 |

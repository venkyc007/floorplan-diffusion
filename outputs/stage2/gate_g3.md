# Gate G3: overfit test, constraints off vs on

Tiny diffusion ([16, 32, 32, 32]), 64 fixed real windows, 400 steps, Adam 3e-3, batch 16, seed 0; ON = all active constraints with the calibrated weights, modes from G2, ramp 50 steps. Scores after 20-step stochastic DDIM (seed 0) inside the holes.

| variant | rec loss (last batch) | hole pixel acc | hole mIoU | minutes |
|---|---|---|---|---|
| off | 0.0303 | 59.9 | 29.1 | 0.7 |
| on | 0.0303 | 53.8 | 25.3 | 2.5 |

ON run: weighted constraint loss and its share of the parameter gradient (|| grad of sum_i w_i L_i || / || grad of the total loss ||), every 50 steps:

| step | rec loss | weighted constraint loss | gradient share |
|---|---|---|---|
| 50 | 0.1068 | 0.00447 | 0.091 |
| 100 | 0.0798 | 0.00219 | 0.059 |
| 150 | 0.0437 | 0.00367 | 0.052 |
| 200 | 0.0392 | 0.00174 | 0.041 |
| 250 | 0.0323 | 0.00255 | 0.073 |
| 300 | 0.0281 | 0.00201 | 0.063 |
| 350 | 0.0276 | 0.00233 | 0.045 |
| 400 | 0.0303 | 0.00144 | 0.063 |

Difference ON - OFF: accuracy -6.1 points, mIoU -3.8 points -> **FAIL** (limit -2 points).

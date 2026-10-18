# Fast (live app) vs full (evaluation) FILL settings, CPU

30 val holes (fixed windows, seed 7; kinds: door 4, fixture 6, railing 2, region 7, room 5, wall 1, window 5), guidance eta 0.3. full: N = 4, steps <= t0 of a 50-step schedule, guidance at every t <= 600; fast: N = 2, 15 steps from t0, guidance on the last 5 steps. Both from the same U-Net sketch.

| mode | hole pixel acc | hole mIoU | violations (chosen sample) | s / hole (mean) | s / hole (median) |
|---|---|---|---|---|---|
| full | 59.3 | 38.1 | 2.80 | 194 | 214 |
| fast | 59.5 | 39.1 | 3.23 | 40 | 43 |

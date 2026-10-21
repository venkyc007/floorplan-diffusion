# Large holes: three fills compared (B1 big window, B2 coarse-to-fine, B3 tiled)

20 fresh held-out val drawings (batch 3's 8 large-kind cases + 6 new medium + 6 new large; measured bins: 7 medium,
13 large). Kaggle CPU, fast sampling, eta 0. Full tables per case: large_fill_compare_manual.md (hole given) and
large_fill_compare_auto.md (full pipeline); figure: large_fill_compare.png (manual mode, the first 2 cases of each
group, no selection; columns input | B1 | B2 (t0 150) | B3 | ground truth; red = the hole box).

## Hole given (manual mode): fill quality alone

| strategy | fixed (IoU >= 0.3) | tol. mIoU in the hole (parse -> fill) | valid (parse -> fill) | s per hole |
|---|---|---|---|---|
| B1 big window | 5 / 20 | 27.9 -> 34.5 | 0% -> 15% | 21 |
| B2 coarse-to-fine, t0 150 | 3 / 20 | 27.9 -> 31.7 | 0% -> 0% | 42 |
| B2 coarse-to-fine, t0 300 | 3 / 20 | 27.9 -> 31.7 | 0% -> 5% | 42 |
| B3 tiled | 6 / 20 | 27.9 -> 33.9 | 0% -> 15% | 18 |

By size bin (fixed / tol. mIoU): medium (7): B1 1 / 28.1, B2 0 / 26.5, B3 1 / 28.1; large (13): B1 4 / 37.9,
B2 3 / 34.5, B3 5 / 37.0.

## Full pipeline (auto mode, CHECK with R8)

Found 15 / 20 (a flag of some type on the omission), but **R8 fired on none of the 20** (region flag 0 / 20), so no
large-hole fill ran and the three strategies give identical results: fixed 2 / 20, tolerant mIoU 27.9 -> 28.8,
validity 0% -> 0%, 21 s per drawing. Why (checked on batch 3 #4): the parser merges the erased bedroom into one
6,628 px "bedroom" (true room 3,358 px), but the bedroom p99 is 10,296 px, so the merged room is only 0.64 x p99.
p99 is about 3x a typical room, so one room merged into its neighbour rarely crosses it, and none of the 20 had a
fixture mix. R8 found 12% of regions on the tuning drawings, 0% here.

## Verdicts (figure cases, manual mode)

| # | case | verdict |
|---|---|---|
| 1 | 3691 room (whole), large | B1 and B3 put the dividing walls back roughly (IoU 0.42 / 0.41: fixed); B2 moves the outer wall inward. None gets the room layout (two bedrooms + hall) right |
| 2 | 14935 room (whole), large | all three draw one bathroom-like room with a fixture blob; the hall / bathroom split is missing (IoU <= 0.12) |
| 3 | 4773 region, medium | all three keep two big rooms; the small dividing walls and the kitchen corner are not drawn (IoU 0.13-0.15) |
| 4 | 4864 room (whole), medium | the bathroom is not recreated; the area is split between bedroom and living (IoU 0.11-0.17) |
| 5 | 14616 room (whole), large | a 13% bedroom: B1 / B3 fill it with kitchen / bedroom / bathroom blobs; B2 adds walls on wrong lines (IoU 0.14-0.18) |
| 6 | 5080 room (whole), large | a 13% kitchen-living: all three cut it with a wall that is not there and mix room types (tol. mIoU 4-14) |

## Conclusion

None of B1-B3 is clearly good: at best 6 / 20 fixed (B3), +6 points of tolerant mIoU over the parse, 15% valid.
B1 and B3 are close (B3 slightly more fixed, B1 slightly higher mIoU). B2 is worse and twice as slow: the Stage 1
layout, mapped back from a ~2x smaller frame, shifts walls by several px, and the refinement at t0 150 / 300 cannot
move them. Typical failures of all three: one big room filled with mixed room-type blobs, dividing walls missing or
on wrong lines, fixtures as blobs. The models were trained on holes of <= ~100 px around ONE component; a missing
room with its walls is outside what they learned. Provisional setting: fill.large_strategy = tiled (most fixed,
fastest). Detection is the other gap: R8 as tuned does not fire on held-out merges.

## B4 plan (not launched): fine-tune at 256 x 256 with large holes

Goal: teach the 16-class U-Net (sketch) and diffusion to draw a whole room / region (walls, doors, room types,
fixtures) inside a 256 px window.

1. Data (CPU, ~1 h on Kaggle CPU): from the completion training set (walls 3 px), 256 x 256 windows around a hole,
   made at label level (no drawing needed): room (whole) (the room + its interior walls / doors / fixtures, 2-15% of
   the building) and region (rectangle, 5-25%), the same definitions as erasure_large.py; 50% large holes, 30% the
   current small kinds, 20% intact windows with random small boxes (keeps the small-hole skill). Mask = the hole box
   (as manual mode) in half the samples, the exact area in the other half. ~40k windows train, 2k val.
2. Models: warm start from k2a_unet and k1_constrained (fully convolutional, no size change needed); all constraint
   losses on, weights as now (calibrated by gradient size). Batch 16 at 256 px (memory like 64 at 128 px).
3. Steps: U-Net 4k, diffusion 6k, on Kaggle T4 x2 (one model per GPU), checkpoint + auto-resume.
4. GPU estimate: a 256 px window costs ~4x a 128 px one (attention maps 32 x 32 instead of 16 x 16 cost a bit more):
   ~16 samples / s per run instead of ~66 on two GPUs. Diffusion 6k x 16 = 96k samples, ~1.7-2 h on one GPU; U-Net
   ~0.7 h on the other. Both in parallel: ~2 h wall time, ~3.5 GPU h; + ~0.5 GPU h for a val check at 256 px.
   Total ~4 GPU h. About 4.3 GPU h are left this week (estimate): it only just fits; after the Saturday reset it is safe.
5. Evaluate with this comparison (manual + auto) on the same 20 cases and a fresh set; keep B4 only if it beats B3
   on fixed and tolerant mIoU without hurting small holes (re-run e2e batch 2 as a check).
6. Detection has to be fixed too, or B4 only helps in manual mode: R8's size test needs a per-drawing reference
   (e.g. a room > 1.6x the median room of its drawing, or > p90 of its type), re-tuned on tuning drawings (CPU).
Risk: the deadline (Sun 25) and the user's absence Wed-Sat; B4 would run unattended after the reset, with the
current setting (tiled) as the fallback.

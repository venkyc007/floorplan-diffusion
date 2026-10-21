# Large holes: three fills compared_manual

20 held-out val drawings: batch 3 8, new medium 6, new large 6. Size bins: small 0, medium 7, large 13. Fast sampling, eta 0, CPU. Scores inside the omission area's box at the parsed map's scale (see the script docstring). Manual = the hole box given (fill quality alone); auto = the full pipeline with R8.

## manual mode: totals

| strategy | fixed | tol. mIoU (parse -> fill) | valid (parse -> fill) | s per hole |
|---|---|---|---|---|
| B1 big window | 5 / 20 | 27.9 -> 34.5 | 0% -> 15% | 21 |
| B2 coarse, t0 150 | 3 / 20 | 27.9 -> 31.7 | 0% -> 0% | 42 |
| B2 coarse, t0 300 | 3 / 20 | 27.9 -> 31.7 | 0% -> 5% | 42 |
| B3 tiled | 6 / 20 | 27.9 -> 33.9 | 0% -> 15% | 18 |

By size bin (manual; fixed / tol. mIoU after per strategy):

| size bin | n | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|
| medium | 7 | 1 / 28.1 | 0 / 26.5 | 0 / 26.7 | 1 / 28.1 |
| large | 13 | 4 / 37.9 | 3 / 34.5 | 3 / 34.4 | 5 / 37.0 |

Per case (manual): fixed (wall IoU) / tol. mIoU per strategy

| # | drawing | group | omission | size | parse tol. | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | batch 3 | room (whole) | large (9.072%) | 42.3 | yes (0.42) / 35.8 | no (0.23) / 22.4 | no (0.23) / 22.2 | yes (0.41) / 30.7 |
| 2 | colorful-14935 | batch 3 | room (whole) | large (7.86%) | 16.6 | no (0.06) / 8.9 | no (0.09) / 10.6 | no (0.09) / 10.6 | no (0.12) / 8.5 |
| 3 | high_quality_architectural-8154 | batch 3 | room (whole) | medium (6.441%) | 22.6 | no (0.00) / 8.7 | no (0.00) / 8.8 | no (0.00) / 8.8 | no (0.00) / 8.5 |
| 4 | colorful-6226 | batch 3 | room (whole) | large (14.473%) | 31.1 | no (0.23) / 82.1 | yes (0.54) / 88.1 | yes (0.54) / 88.1 | no (0.29) / 82.9 |
| 5 | high_quality_architectural-1376 | batch 3 | region | large (24.718%) | 9.0 | no (0.24) / 18.5 | yes (0.37) / 19.0 | yes (0.37) / 19.3 | no (0.25) / 20.4 |
| 6 | colorful-14938 | batch 3 | region | large (11.17%) | 46.5 | yes (0.35) / 56.7 | no (0.28) / 43.8 | no (0.28) / 43.1 | yes (0.38) / 57.5 |
| 7 | colorful-6628 | batch 3 | region | large (13.917%) | 68.8 | yes (0.54) / 60.2 | no (0.07) / 31.7 | no (0.07) / 31.6 | yes (0.51) / 48.2 |
| 8 | high_quality-9975 | batch 3 | region | large (19.228%) | 34.1 | no (0.13) / 29.3 | no (0.28) / 35.2 | no (0.28) / 34.5 | no (0.19) / 32.8 |
| 9 | high_quality-4773 | new medium | region | medium (5.192%) | 16.4 | no (0.13) / 15.6 | no (0.15) / 16.5 | no (0.15) / 16.5 | no (0.13) / 15.7 |
| 10 | high_quality_architectural-4864 | new medium | room (whole) | medium (4.608%) | 6.7 | no (0.17) / 22.5 | no (0.14) / 13.4 | no (0.14) / 13.9 | no (0.11) / 19.1 |
| 11 | high_quality-6732 | new medium | region | medium (6.013%) | 15.1 | no (0.18) / 12.8 | no (0.17) / 11.6 | no (0.17) / 12.3 | no (0.18) / 12.4 |
| 12 | high_quality-4550 | new medium | region | medium (5.593%) | 23.9 | no (0.19) / 30.6 | no (0.14) / 34.8 | no (0.14) / 34.8 | no (0.25) / 36.3 |
| 13 | high_quality_architectural-7864 | new medium | region | medium (7.468%) | 39.6 | no (0.29) / 100.0 | no (0.23) / 98.7 | no (0.23) / 98.7 | no (0.29) / 100.0 |
| 14 | high_quality-7082 | new medium | region | medium (5.227%) | 8.6 | yes (0.33) / 6.4 | no (0.21) / 1.9 | no (0.21) / 1.9 | yes (0.33) / 4.7 |
| 15 | colorful-14616 | new large | room (whole) | large (12.836%) | 27.4 | no (0.16) / 27.6 | no (0.18) / 19.9 | no (0.18) / 19.4 | no (0.14) / 22.8 |
| 16 | high_quality-5080 | new large | room (whole) | large (12.604%) | 6.9 | no (0.15) / 4.2 | no (0.16) / 13.6 | no (0.16) / 13.4 | no (0.16) / 5.0 |
| 17 | high_quality_architectural-9438 | new large | region | large (12.153%) | 69.6 | no (0.17) / 79.2 | no (0.20) / 85.2 | no (0.20) / 85.3 | no (0.10) / 87.6 |
| 18 | high_quality_architectural-2080 | new large | region | large (14.294%) | 9.6 | no (0.22) / 8.1 | no (0.27) / 10.7 | no (0.27) / 10.5 | no (0.22) / 11.6 |
| 19 | high_quality_architectural-1324 | new large | room (whole) | large (13.864%) | 36.9 | no (0.29) / 57.3 | yes (0.36) / 47.8 | yes (0.36) / 47.9 | yes (0.34) / 53.4 |
| 20 | high_quality_architectural-8796 | new large | room (whole) | large (12.743%) | 25.7 | yes (0.32) / 25.3 | no (0.26) / 20.7 | no (0.26) / 20.7 | yes (0.33) / 19.6 |


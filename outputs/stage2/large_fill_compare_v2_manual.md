# Large holes: three fills compared_v2_manual

20 held-out val drawings: batch 3 8, new medium 6, new large 6. Size bins: small 0, medium 7, large 13. Fast sampling, eta 0, CPU. Scores inside the omission area's box at the parsed map's scale (see the script docstring). Manual = the hole box given (fill quality alone); auto = the full pipeline with R8.

## manual mode: totals

| strategy | fixed | tol. mIoU (parse -> fill) | valid (parse -> fill) | s per hole |
|---|---|---|---|---|
| B1 big window | 6 / 20 | 27.9 -> 32.1 | 0% -> 15% | 22 |
| B2 coarse, t0 150 | 3 / 20 | 27.9 -> 31.5 | 0% -> 5% | 44 |
| B2 coarse, t0 300 | 3 / 20 | 27.9 -> 31.7 | 0% -> 5% | 44 |
| B3 tiled | 5 / 20 | 27.9 -> 33.3 | 0% -> 15% | 19 |

By size bin (manual; fixed / tol. mIoU after per strategy):

| size bin | n | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|
| medium | 7 | 0 / 30.5 | 0 / 28.8 | 0 / 29.0 | 0 / 29.9 |
| large | 13 | 6 / 32.9 | 3 / 33.0 | 3 / 33.1 | 5 / 35.1 |

Per case (manual): fixed (wall IoU) / tol. mIoU per strategy

| # | drawing | group | omission | size | parse tol. | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | batch 3 | room (whole) | large (9.072%) | 42.3 | yes (0.37) / 35.1 | no (0.23) / 23.5 | no (0.23) / 23.4 | yes (0.44) / 27.4 |
| 2 | colorful-14935 | batch 3 | room (whole) | large (7.86%) | 16.6 | no (0.08) / 7.0 | no (0.08) / 10.4 | no (0.08) / 9.8 | no (0.10) / 7.2 |
| 3 | high_quality_architectural-8154 | batch 3 | room (whole) | medium (6.441%) | 22.6 | no (0.00) / 22.1 | no (0.00) / 19.4 | no (0.00) / 21.0 | no (0.00) / 12.3 |
| 4 | colorful-6226 | batch 3 | room (whole) | large (14.473%) | 31.1 | yes (0.35) / 67.7 | yes (0.54) / 87.9 | yes (0.54) / 88.1 | no (0.26) / 77.5 |
| 5 | high_quality_architectural-1376 | batch 3 | region | large (24.718%) | 9.0 | no (0.23) / 13.0 | yes (0.37) / 16.2 | yes (0.37) / 16.2 | no (0.23) / 12.9 |
| 6 | colorful-14938 | batch 3 | region | large (11.17%) | 46.5 | yes (0.36) / 56.7 | no (0.28) / 43.0 | no (0.28) / 43.0 | yes (0.38) / 58.0 |
| 7 | colorful-6628 | batch 3 | region | large (13.917%) | 68.8 | no (0.04) / 56.3 | no (0.08) / 31.8 | no (0.05) / 31.8 | no (0.16) / 57.7 |
| 8 | high_quality-9975 | batch 3 | region | large (19.228%) | 34.1 | no (0.15) / 29.3 | no (0.28) / 31.3 | no (0.28) / 31.6 | no (0.19) / 35.1 |
| 9 | high_quality-4773 | new medium | region | medium (5.192%) | 16.4 | no (0.13) / 15.9 | no (0.15) / 16.7 | no (0.15) / 16.7 | no (0.13) / 15.8 |
| 10 | high_quality_architectural-4864 | new medium | room (whole) | medium (4.608%) | 6.7 | no (0.11) / 16.1 | no (0.13) / 13.0 | no (0.12) / 12.9 | no (0.12) / 19.2 |
| 11 | high_quality-6732 | new medium | region | medium (6.013%) | 15.1 | no (0.18) / 17.6 | no (0.17) / 16.1 | no (0.18) / 19.3 | no (0.18) / 20.3 |
| 12 | high_quality-4550 | new medium | region | medium (5.593%) | 23.9 | no (0.15) / 35.7 | no (0.14) / 35.4 | no (0.14) / 32.2 | no (0.25) / 35.7 |
| 13 | high_quality_architectural-7864 | new medium | region | medium (7.468%) | 39.6 | no (0.29) / 100.0 | no (0.23) / 98.7 | no (0.23) / 98.7 | no (0.29) / 100.0 |
| 14 | high_quality-7082 | new medium | region | medium (5.227%) | 8.6 | no (0.25) / 6.2 | no (0.21) / 2.2 | no (0.21) / 2.2 | no (0.25) / 6.1 |
| 15 | colorful-14616 | new large | room (whole) | large (12.836%) | 27.4 | no (0.14) / 24.1 | no (0.18) / 25.6 | no (0.18) / 26.6 | no (0.12) / 28.9 |
| 16 | high_quality-5080 | new large | room (whole) | large (12.604%) | 6.9 | no (0.15) / 6.4 | no (0.16) / 12.6 | no (0.16) / 13.9 | no (0.15) / 5.5 |
| 17 | high_quality_architectural-9438 | new large | region | large (12.153%) | 69.6 | yes (0.38) / 54.1 | no (0.20) / 65.6 | no (0.20) / 65.5 | yes (0.38) / 71.3 |
| 18 | high_quality_architectural-2080 | new large | region | large (14.294%) | 9.6 | no (0.15) / 8.1 | no (0.28) / 10.4 | no (0.28) / 9.0 | no (0.20) / 6.3 |
| 19 | high_quality_architectural-1324 | new large | room (whole) | large (13.864%) | 36.9 | yes (0.38) / 49.5 | yes (0.36) / 47.7 | yes (0.36) / 47.8 | yes (0.40) / 48.2 |
| 20 | high_quality_architectural-8796 | new large | room (whole) | large (12.743%) | 25.7 | yes (0.36) / 20.1 | no (0.26) / 23.2 | no (0.26) / 23.7 | yes (0.34) / 19.7 |


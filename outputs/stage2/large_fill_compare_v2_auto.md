# Large holes: three fills compared_v2_auto

20 held-out val drawings: batch 3 8, new medium 6, new large 6. Size bins: small 0, medium 7, large 13. Fast sampling, eta 0, CPU. Scores inside the omission area's box at the parsed map's scale (see the script docstring). Manual = the hole box given (fill quality alone); auto = the full pipeline with R8.

## auto mode: totals

| strategy | found | region flag | fixed | tol. mIoU (parse -> fill) | valid (parse -> fill) | s per drawing |
|---|---|---|---|---|---|---|
| B1 big window | 15 / 20 | 0 / 20 | 2 / 20 | 27.9 -> 29.2 | 0% -> 0% | 20 |
| B2 coarse, t0 150 | 15 / 20 | 0 / 20 | 2 / 20 | 27.9 -> 29.2 | 0% -> 0% | 20 |
| B2 coarse, t0 300 | 15 / 20 | 0 / 20 | 2 / 20 | 27.9 -> 29.2 | 0% -> 0% | 20 |
| B3 tiled | 15 / 20 | 0 / 20 | 2 / 20 | 27.9 -> 29.2 | 0% -> 0% | 20 |

By size bin (auto; fixed / tol. mIoU after per strategy):

| size bin | n | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|
| medium | 7 | 0 / 19.1 | 0 / 19.1 | 0 / 19.1 | 0 / 19.1 |
| large | 13 | 2 / 34.7 | 2 / 34.7 | 2 / 34.7 | 2 / 34.7 |

Per case (auto): fixed (wall IoU) / tol. mIoU per strategy

| # | drawing | group | omission | size | flags | parse tol. | B1 big window | B2 coarse, t0 150 | B2 coarse, t0 300 | B3 tiled |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | high_quality_architectural-3691 | batch 3 | room (whole) | large (9.072%) | window (WINK+detector); room (detector); room (INK+detector) | 42.3 | no (0.14) / 43.1 | no (0.14) / 43.1 | no (0.14) / 43.1 | no (0.14) / 43.1 |
| 2 | colorful-14935 | batch 3 | room (whole) | large (7.86%) | sink (detector) | 16.6 | no (0.03) / 16.0 | no (0.03) / 16.0 | no (0.03) / 16.0 | no (0.03) / 16.0 |
| 3 | high_quality_architectural-8154 | batch 3 | room (whole) | medium (6.441%) | none | 22.6 | no (0.00) / 22.6 | no (0.00) / 22.6 | no (0.00) / 22.6 | no (0.00) / 22.6 |
| 4 | colorful-6226 | batch 3 | room (whole) | large (14.473%) | room (detector) | 31.1 | yes (0.46) / 32.7 | yes (0.46) / 32.7 | yes (0.46) / 32.7 | yes (0.46) / 32.7 |
| 5 | high_quality_architectural-1376 | batch 3 | region | large (24.718%) | door (R5); room (detector) | 9.0 | no (0.03) / 9.5 | no (0.03) / 9.5 | no (0.03) / 9.5 | no (0.03) / 9.5 |
| 6 | colorful-14938 | batch 3 | region | large (11.17%) | window (detector); sink (detector) | 46.5 | no (0.02) / 46.5 | no (0.02) / 46.5 | no (0.02) / 46.5 | no (0.02) / 46.5 |
| 7 | colorful-6628 | batch 3 | region | large (13.917%) | door (INK) | 68.8 | no (0.00) / 68.8 | no (0.00) / 68.8 | no (0.00) / 68.8 | no (0.00) / 68.8 |
| 8 | high_quality-9975 | batch 3 | region | large (19.228%) | door (INK) | 34.1 | no (0.01) / 34.1 | no (0.01) / 34.1 | no (0.01) / 34.1 | no (0.01) / 34.1 |
| 9 | high_quality-4773 | new medium | region | medium (5.192%) | door (detector) | 16.4 | no (0.04) / 17.4 | no (0.04) / 17.4 | no (0.04) / 17.4 | no (0.04) / 17.4 |
| 10 | high_quality_architectural-4864 | new medium | room (whole) | medium (4.608%) | sink (detector) | 6.7 | no (0.02) / 6.7 | no (0.02) / 6.7 | no (0.02) / 6.7 | no (0.02) / 6.7 |
| 11 | high_quality-6732 | new medium | region | medium (6.013%) | door (R5); door (INK+detector); window (WINK) | 15.1 | no (0.04) / 15.1 | no (0.04) / 15.1 | no (0.04) / 15.1 | no (0.04) / 15.1 |
| 12 | high_quality-4550 | new medium | region | medium (5.593%) | none | 23.9 | no (0.02) / 23.9 | no (0.02) / 23.9 | no (0.02) / 23.9 | no (0.02) / 23.9 |
| 13 | high_quality_architectural-7864 | new medium | region | medium (7.468%) | none | 39.6 | no (0.03) / 39.6 | no (0.03) / 39.6 | no (0.03) / 39.6 | no (0.03) / 39.6 |
| 14 | high_quality-7082 | new medium | region | medium (5.227%) | door (detector); sink (detector) | 8.6 | no (0.21) / 8.6 | no (0.21) / 8.6 | no (0.21) / 8.6 | no (0.21) / 8.6 |
| 15 | colorful-14616 | new large | room (whole) | large (12.836%) | room (detector) | 27.4 | yes (0.46) / 23.9 | yes (0.46) / 23.9 | yes (0.46) / 23.9 | yes (0.46) / 23.9 |
| 16 | high_quality-5080 | new large | room (whole) | large (12.604%) | none | 6.9 | no (0.17) / 6.9 | no (0.17) / 6.9 | no (0.17) / 6.9 | no (0.17) / 6.9 |
| 17 | high_quality_architectural-9438 | new large | region | large (12.153%) | door (R5); door (R5+detector); room (INK+detector); room (detector) | 69.6 | no (0.15) / 89.0 | no (0.15) / 89.0 | no (0.15) / 89.0 | no (0.15) / 89.0 |
| 18 | high_quality_architectural-2080 | new large | region | large (14.294%) | door (detector); window (WINK+detector) | 9.6 | no (0.03) / 9.6 | no (0.03) / 9.6 | no (0.03) / 9.6 | no (0.03) / 9.6 |
| 19 | high_quality_architectural-1324 | new large | room (whole) | large (13.864%) | window (detector) | 36.9 | no (0.09) / 36.9 | no (0.09) / 36.9 | no (0.09) / 36.9 | no (0.09) / 36.9 |
| 20 | high_quality_architectural-8796 | new large | room (whole) | large (12.743%) | door (R5); window (WINK+detector) | 25.7 | no (0.05) / 33.9 | no (0.05) / 33.9 | no (0.05) / 33.9 | no (0.05) / 33.9 |


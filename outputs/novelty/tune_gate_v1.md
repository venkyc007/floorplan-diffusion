# Region flags kept only if the fill would change the area_ffc

Locator = the configured one (average if several), blob >= 150 px. change = wall pixels the large-hole U-Net draws in the flagged box that are not within 2 px of a wall of the parsed map. found = an omission with a kept flag on it; flags / intact = kept flags per intact drawing.

| threshold | min change (wall px) | found | found, medium | found, large | flags / intact drawing | intact drawings flagged | stray flags / erased drawing |
|---|---|---|---|---|---|---|---|
| 0.3 | 0 | 75% | 72% | 76% | 0.18 | 17% | 0.08 |
| 0.3 | 10 | 74% | 70% | 75% | 0.17 | 15% | 0.06 |
| 0.3 | 20 | 71% | 69% | 71% | 0.16 | 15% | 0.06 |
| 0.3 | 40 | 65% | 62% | 67% | 0.14 | 13% | 0.05 |
| 0.3 | 80 | 46% | 43% | 48% | 0.09 | 8% | 0.03 |
| 0.3 | 160 | 22% | 20% | 23% | 0.03 | 3% | 0.01 |
| 0.5 | 0 | 67% | 64% | 68% | 0.11 | 10% | 0.04 |
| 0.5 | 10 | 66% | 63% | 67% | 0.09 | 9% | 0.04 |
| 0.5 | 20 | 64% | 62% | 65% | 0.09 | 8% | 0.03 |
| 0.5 | 40 | 58% | 55% | 59% | 0.07 | 7% | 0.03 |
| 0.5 | 80 | 41% | 36% | 43% | 0.04 | 4% | 0.01 |
| 0.5 | 160 | 20% | 17% | 21% | 0.01 | 1% | 0.01 |
| 0.7 | 0 | 66% | 64% | 68% | 0.10 | 10% | 0.03 |
| 0.7 | 10 | 65% | 62% | 67% | 0.08 | 8% | 0.03 |
| 0.7 | 20 | 64% | 62% | 64% | 0.08 | 8% | 0.03 |
| 0.7 | 40 | 56% | 51% | 58% | 0.07 | 7% | 0.02 |
| 0.7 | 80 | 39% | 35% | 42% | 0.03 | 3% | 0.01 |
| 0.7 | 160 | 19% | 15% | 20% | 0.01 | 1% | 0.01 |

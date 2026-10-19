# C8 on parsed drawings: why cut-off rooms are missed

15 held-out drawings with a door whose closure cuts a room off (door erased as kind (a), erasure v2, parser from configs/stage2/pipeline.yaml) + extra drawings. Categories: see src/stage2/detect/c8_diag.py.

| drawing | outcome | C8 holes in drawing |
|---|---|---|
| colorful-13513 | below C8 area threshold (938 < 2756 px) | 0 |
| colorful-13689 | reached through another parsed door | 0 |
| colorful-13804 | below C8 area threshold (1479 < 2756 px) | 0 |
| colorful-13820 | flagged | 1 |
| colorful-13824 | reached through another parsed door | 0 |
| colorful-13826 | reached through another parsed door | 0 |
| colorful-14126 | flagged | 1 |
| colorful-14251 | below C8 area threshold (1324 < 2756 px) | 0 |
| colorful-14328 | below C8 area threshold (1479 < 2756 px) | 0 |
| colorful-14616 | door redrawn | 0 |
| colorful-14928 | below C8 area threshold (498 < 2756 px) | 0 |
| colorful-14935 | below C8 area threshold (366 < 2756 px) | 0 |
| colorful-14936 | door redrawn | 0 |
| colorful-14938 | below C8 area threshold (259 < 2756 px) | 0 |
| colorful-3134 | below C8 area threshold (1290 < 2756 px) | 0 |
| colorful-1261 (extra) | flagged | 1 |

Totals: below C8 area threshold (1290 < 2756 px) 1, below C8 area threshold (1324 < 2756 px) 1, below C8 area threshold (1479 < 2756 px) 2, below C8 area threshold (259 < 2756 px) 1, below C8 area threshold (366 < 2756 px) 1, below C8 area threshold (498 < 2756 px) 1, below C8 area threshold (938 < 2756 px) 1, door redrawn 2, flagged 3, reached through another parsed door 3.

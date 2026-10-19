# Detectors v1 / v2 / v3 on the same validation crops (parsed TUNING drawings, 384 px, parser scale)

512 crops. Detected = >= 30% of an omitted component's pixels predicted as its type; false alarms = intact crops with >= 20 predicted missing pixels. v1 / v2 see the parse at completion scale.

| detector | mean F1 | false alarms (intact crops) | F1 / detected wall | F1 / detected door | F1 / detected window | F1 / detected railing | F1 / detected room | F1 / detected toilet | F1 / detected sink | F1 / detected bath_shower | F1 / detected appliance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 (drawing + parse) | 0.560 | 0.11 | 0.89 / 0.97 | 0.74 / 0.81 | 0.79 / 0.69 | 0.66 / 0.79 | 0.82 / 0.89 | 0.10 / 1.00 | 0.28 / 0.67 | 0.03 / 0.09 | 0.74 / 1.00 |
| v1 (label-level) | 0.048 | 0.88 | 0.00 / 0.05 | 0.07 / 0.10 | 0.14 / 0.40 | 0.01 / 0.04 | 0.02 / 0.07 | 0.00 / 0.00 | 0.17 / 0.61 | 0.00 / 0.00 | 0.02 / 0.29 |
| v2 (parsed maps) | 0.068 | 0.48 | 0.01 / 0.00 | 0.10 / 0.08 | 0.26 / 0.38 | 0.00 / 0.00 | 0.16 / 0.20 | 0.00 / 0.00 | 0.05 / 0.06 | 0.00 / 0.00 | 0.03 / 0.07 |

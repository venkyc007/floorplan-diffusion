# Validity of what is added (medium and large omissions)

Counts on the walls / pieces a plan adds inside the omission's box (src/novelty/regularise.checks); rule violations and `valid` are the project's own rule check (src/stage2/pipeline/e2e_val.py: `valid` = not more violations than the true layout).

## medium omissions

| arm | n | recovered (2 px) | fixed (strict) | rule violations (truth) | valid | added wall px | not straight | thickness spread, px | loose wall groups | specks | small room patches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid v3c auto | 208 | 67% | 51% | 16.7 (7.0) | 9% | 183 | 2.0% | 0.53 | 0.34 | 1.06 | 0.18 |
| hybrid v3d-noend auto | 208 | 63% | 48% | 19.5 (7.0) | 6% | 196 | 0.7% | 0.22 | 0.04 | 0.06 | 0.26 |
| hybrid v3d auto | 208 | 60% | 47% | 19.1 (7.0) | 6% | 199 | 0.7% | 0.22 | 0.02 | 0.07 | 0.27 |
| true layout | 208 | - | - | 7.0 | 100% | 117 | 2.9% | 0.42 | 0.09 | 1.10 | 0.03 |

## large omissions

| arm | n | recovered (2 px) | fixed (strict) | rule violations (truth) | valid | added wall px | not straight | thickness spread, px | loose wall groups | specks | small room patches |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid v3c auto | 208 | 74% | 53% | 24.3 (11.2) | 12% | 301 | 1.9% | 0.67 | 0.62 | 1.59 | 0.27 |
| hybrid v3d-noend auto | 208 | 73% | 46% | 28.0 (11.2) | 6% | 305 | 0.6% | 0.25 | 0.03 | 0.05 | 0.40 |
| hybrid v3d auto | 208 | 64% | 46% | 27.8 (11.2) | 6% | 303 | 0.6% | 0.23 | 0.02 | 0.10 | 0.39 |
| true layout | 208 | - | - | 11.2 | 100% | 184 | 3.2% | 0.58 | 0.23 | 1.69 | 0.04 |


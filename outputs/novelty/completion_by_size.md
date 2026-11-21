# Completion rate by omission size

## What counts as a completed omission

One omission is erased from a real drawing; the system gets only the damaged drawing (no mask, no hint).

- **found**: the CHECK stage puts a flag (a marked box) on the erased spot.
- **completed ("fixed", the 46% figure)**: found, AND the final map has the erased component's class back at the
  erased spot with IoU >= 0.3 between the erased pixels and the plan's pixels of that class there (whoever drew
  them: the fill or the parser). For a wall-type omission (wall piece, a room's walls, whole room, region) the class
  is wall.
- **recovered (2 px)** (large omissions, novelty work): found, AND at least half of the erased wall pixels have a
  wall of the plan within 2 px, AND at least half of the walls the plan adds there lie within 2 px of a true wall.
  Used because walls are 3 px thick, so a wall drawn one or two pixels off scores almost no IoU; the strict rule
  above tops out at 72% even when the true layout is pasted back (parser maps are not pixel-exact).
- **sound** (round 4, see `soundness_*.md`): the completed region passes the three structural checks (walls
  connect, openings sit in walls, rooms closed and reachable). Judged without the true layout, so a plausible
  alternative to the true layout can pass.

Sizes: omitted area as a share of the building's area: small < 2.7%, medium 2.7-7.7%, large >= 7.7% (the
project's bins, src/stage2/data/omission_meta.py). Brackets: Wilson 95% intervals.

## 1. The 46%: Stage 2 TEST, one single-component omission per drawing (2026-10-22)

330 test drawings (railings left out: the system handles them only in manual mode). Single components are almost always small, so the small bin is split by area.

| omissions | n | found | completed (fixed) | drawn by the fill |
|---|---|---|---|---|
| all except railings (the 46%) | 330 | 69% [64, 74] | 46% [41, 52] | 25% [21, 30] |
| small, < 0.25% of the building | 157 | 61% [53, 68] | 39% [32, 47] | 17% [12, 23] |
| small, 0.25-1% | 142 | 75% [67, 81] | 51% [43, 59] | 32% [25, 40] |
| small, 1-2.7% | 29 | 83% [65, 92] | 62% [44, 77] | 34% [20, 53] |
| medium, 2.7-7.7% | 2 | 100% [34, 100] | 50% [9, 91] | 50% [9, 91] |
| large, >= 7.7% | 0 | - | - | - |

By kind and size (completed / n):

| kind | < 0.25% | 0.25-1% | 1-2.7% | >= 2.7% |
|---|---|---|---|---|
| door (a) | 13/41 | 7/11 | 0/2 | - |
| door (b) | 19/38 | 5/14 | - | - |
| fixture | 4/30 | 7/30 | 0/3 | 0/1 |
| room | 1/4 | 23/36 | 9/12 | 1/1 |
| wall | 17/31 | 13/21 | 1/2 | - |
| window | 7/13 | 18/30 | 8/10 | - |

## 2. Stage 2 TEST, size-balanced end to end (2026-10-22, shipped pipeline)

120 test drawings, a third each small / medium / large omissions of mixed kinds (doors, windows, walls, fixtures, railings, whole rooms, erased regions).

| omissions | n | found | completed (fixed) | drawn by the fill |
|---|---|---|---|---|
| all | 120 | 66% [57, 74] | 12% [8, 20] | 7% [3, 13] |
| small | 40 | 50% [35, 65] | 22% [12, 38] | 8% [3, 20] |
| medium | 40 | 68% [52, 80] | 15% [7, 29] | 10% [4, 23] |
| large | 40 | 80% [65, 90] | 0% [0, 9] | 2% [0, 13] |

## 3. Large omissions after the novelty work (held-out VALIDATION drawings, 151, test not used)

Same definitions; one small, one medium and one large omission per drawing (separate runs).

**shipped pipeline**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) |
|---|---|---|---|---|
| small | 151 | 61% [53, 68] | 42% [34, 50] | 50% [31, 69] (n 22) |
| medium | 151 | 73% [65, 79] | 15% [10, 21] | 17% [12, 24] |
| large | 151 | 80% [73, 86] | 13% [8, 19] | 11% [7, 17] |

**round 1 hybrid (v2)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) |
|---|---|---|---|---|
| small | 151 | 64% [56, 71] | 42% [34, 50] | 55% [35, 73] (n 22) |
| medium | 151 | 87% [81, 92] | 38% [30, 46] | 50% [42, 58] |
| large | 151 | 95% [91, 98] | 36% [29, 44] | 52% [44, 59] |

**final hybrid v3c (round 2)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) |
|---|---|---|---|---|
| small | 151 | 62% [54, 69] | 41% [34, 49] | 55% [35, 73] (n 22) |
| medium | 151 | 95% [91, 98] | 57% [49, 65] | 79% [72, 85] |
| large | 151 | 99% [95, 100] | 55% [47, 63] | 80% [73, 86] |

**round 4 final v4c (structurally sound fills)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) | sound (3 structural checks) |
|---|---|---|---|---|---|
| small | 151 | 62% [54, 69] | 40% [33, 48] | 55% [35, 73] (n 22) | 64% [43, 80] (n 22) |
| medium | 151 | 95% [91, 98] | 51% [43, 59] | 71% [63, 78] | 83% [77, 89] |
| large | 151 | 99% [95, 100] | 44% [37, 52] | 74% [66, 80] | 81% [74, 86] |


## 4. TEST drawings (each version run once; v4c kernel v17, the others kernel v18, 2026-11-21)

Same definitions and omission generator; every typed test drawing (369-371), the same cases for every version. v4c is the final hybrid (frozen before its run); v3c, v4b and the shipped pipeline were run afterwards for the comparison only (user request), nothing was chosen on them.

**round 4 final v4c (structurally sound fills)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) | sound (3 structural checks) |
|---|---|---|---|---|---|
| small | 371 | 65% [61, 70] | 40% [35, 45] | 45% [32, 59] (n 51) | 51% [38, 64] (n 51) |
| medium | 369 | 88% [85, 91] | 40% [35, 45] | 63% [58, 68] | 85% [81, 88] |
| large | 369 | 98% [96, 99] | 47% [42, 53] | 66% [61, 71] | 78% [74, 82] |

**v4b (most sound on tuning)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) | sound (3 structural checks) |
|---|---|---|---|---|---|
| small | 371 | 65% [61, 70] | 40% [35, 45] | 47% [34, 60] (n 51) | 53% [40, 66] (n 51) |
| medium | 369 | 88% [85, 91] | 40% [35, 45] | 63% [58, 68] | 87% [83, 90] |
| large | 369 | 98% [96, 99] | 49% [43, 54] | 68% [63, 72] | 85% [81, 88] |

**v3c (round 2 final, accuracy first)**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) | sound (3 structural checks) |
|---|---|---|---|---|---|
| small | 371 | 65% [61, 70] | 41% [36, 46] | 51% [38, 64] (n 51) | 45% [32, 59] (n 51) |
| medium | 369 | 88% [85, 91] | 54% [49, 59] | 71% [66, 75] | 37% [32, 42] |
| large | 369 | 98% [96, 99] | 56% [50, 61] | 78% [74, 82] | 19% [15, 23] |

**shipped pipeline**:

| omissions | n | found | completed (fixed, strict) | recovered (2 px) | sound (3 structural checks) |
|---|---|---|---|---|---|
| small | 371 | 65% [60, 70] | 40% [35, 45] | 51% [38, 64] (n 51) | 47% [34, 60] (n 51) |
| medium | 369 | 70% [65, 74] | 11% [8, 14] | 9% [7, 13] | 45% [40, 50] |
| large | 369 | 82% [78, 86] | 10% [8, 14] | 8% [5, 11] | 36% [31, 41] |

Reading: completion falls with size in the shipped system (single small components about 2 in 5, large erased regions almost never). The novelty work targets the large end; small omissions are unchanged by it. Round 4 trades some accuracy for structurally sound fills (`outputs/novelty/soundness_heldout.md`).

# Redraw rate: does the parser (v1) label an ERASED component back? (erasure v2, held-out val)

100 held-out drawings; per type one component erased at a time (whole symbol: door zone / box + 40%). Redrawn = share of the removed component's pixels the parser still labels with its class (mean over cases; pooled over pixels in brackets). Found when intact = the same share on the original drawing.

| type | cases | redrawn | cases > 50% redrawn | found when intact |
|---|---|---|---|---|
| door (a) | 100 | 11% (12%) | 10% | 88% |
| door (b) | 100 | 79% (81%) | 79% | 86% |
| window | 100 | 38% (44%) | 39% | 93% |
| fixture | 100 | 11% (3%) | 11% | 82% |
| door (a + b) | 200 | 45% (47%) | 44% | 87% |

Fixtures by class: appliance 3% (34), bath_shower 39% (13), sink 2% (35), toilet 21% (18).

Before (overnight, erasure v1, image_eval on other omissions): door (a) 55%, door (b) 85%, window 77%, fixture 8%.

# Control: do the detectors fire on erasure TRACES? (fake erasures, intact held-out drawings)

40 intact held-out drawings. Fake erasures of non-components (nothing is missing afterwards): a room-name word (inpainted as text erasure), a furniture drawing and a blank floor patch (both erased as a fixture is). Firing = a detector proposal overlaps the spot; baseline = the same spot on the untouched drawing. Real omissions (held-out, same thresholds): detectors found 58% (detection_eval.md).

| fake erasure | spots | v3 fires | v3 baseline | v2 fires | v2 baseline |
|---|---|---|---|---|---|
| word | 40 | 0% | 0% | 0% | 0% |
| furniture | 13 | 23% | 0% | 0% | 0% |
| blank | 40 | 2% | 2% | 0% | 0% |

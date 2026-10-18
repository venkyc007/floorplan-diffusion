# Pipeline examples (CPU)

## colorful-11261 (door whitened)

CPU time 556 s ({'parse': 6.7, 'detect': 2.1, 'fill': 547.5}); parse scale 1.00, walls 9.6 px.

- wall missing (detector) -> wall, window, living, outdoor added in living
- appliance missing (detector) -> kitchen, other_room, appliance added in kitchen
- bath_shower missing (detector) -> sink added in bathroom

## colorful-1148 (window whitened)

CPU time 8 s ({'parse': 3.9, 'detect': 3.8, 'fill': 0.0}); parse scale 1.00, walls 9.6 px.


## colorful-1261 (fixture whitened)

CPU time 499 s ({'parse': 6.0, 'detect': 7.8, 'fill': 485.3}); parse scale 1.00, walls 7.6 px.

- railing missing (detector) -> railing, outdoor, background added in the plan
- sink missing (detector) -> bathroom, sink added in bathroom
- sink missing (detector) -> toilet, sink added in bathroom

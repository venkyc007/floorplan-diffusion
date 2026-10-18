# Pipeline examples (CPU, fast mode)

Whitening: only the removed component's own pixels (its rasterised polygon), grown 2 px.
## colorful-11261 (door whitened)

CPU time 11 s ({'parse': 8.0, 'detect': 3.1, 'fill': 0.0}); parse scale 1.00, walls 9.6 px.


## colorful-1148 (window whitened)

CPU time 9 s ({'parse': 3.6, 'detect': 5.0, 'fill': 0.0}); parse scale 1.00, walls 9.6 px.


## colorful-1261 (fixture whitened)

CPU time 240 s ({'parse': 5.6, 'detect': 11.4, 'fill': 222.4}); parse scale 1.00, walls 7.6 px.

- bedroom had no door -> door added on the bedroom / other_room wall
- bedroom had no door -> door added on the bedroom / other_room wall
- bedroom had no door -> door added on the bedroom / hallway wall
- bedroom had no door -> door added on the bedroom / hallway wall
- bedroom had no door -> door added on the bedroom / other_room wall
- door missing (C8) -> 6 walls, 1 toilet added in bedroom
- railing missing (detector) -> 1 railing added in the plan
- sink missing (detector) -> 1 sink added in bathroom
- sink missing (detector) -> 1 toilet, 1 sink added in bathroom
- wall missing (detector) -> 2 walls, 1 railing added in other_room

## colorful-13513 (door (room cut off) whitened)

CPU time 3 s ({'parse': 2.1, 'detect': 1.4, 'fill': 0.0}); parse scale 1.40, walls 11.5 px.


## colorful-13689 (door (room cut off) whitened)

CPU time 5 s ({'parse': 2.3, 'detect': 2.7, 'fill': 0.0}); parse scale 1.00, walls 9.6 px.


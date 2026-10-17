# Stage 2 data: CubiCasa5K rasterised from SVG

Built 2026-10-17 by `python -m src.stage2.data.build_cubicasa` on Kaggle (source: Kaggle dataset `qmarva/cubicasa5k`, the complete mirror of Zenodo 2613548). Parser set: greyscale drawing + labels, median wall 8 px, long side <= 2048. Completion set: labels only, median wall 3 px. Official split kept.

## Plans

| split | listed | kept | dropped |
|---|---|---|---|
| train | 4200 | 4200 | 0 |
| val | 400 | 400 | 0 |
| test | 400 | 400 | 0 |
| total | 5000 | 5000 | 0 |

Plan folders with a model.svg: 5000; listed in the official split files: 5000 (0 not in any list, not used).

**Dropped plans and why:**

none

## Scale

| quantity | median | p2 | p98 | max |
|---|---|---|---|---|
| wall thickness in the SVG frame (px) | 19.7 | 11.4 | 35.1 | 78.0 |
| parser map long side (px) | 662 | 315 | 1556 | 2048 |
| completion map long side (px) | 248 | 118 | 584 | 2261 |

Plans that hit the 2048-px cap (walls thinner than 8 px in the parser set): 22; their wall width there: median 7.2 px.
Image source: 525 plans from F1_original.png (higher resolution), 4475 from F1_scaled.png.

## Class pixel frequencies and components (train split)

| class | % pixels (completion) | % pixels (parser) | % plans with it | components (4-conn) | components per plan |
|---|---|---|---|---|---|
| wall | 6.974 | 6.136 | 100.0 | 46949 | 11.18 |
| door | 0.811 | 0.653 | 100.0 | 41237 | 9.82 |
| window | 1.519 | 1.332 | 99.5 | 31514 | 7.50 |
| railing | 0.462 | 0.362 | 74.0 | 7729 | 1.84 |
| living | 7.534 | 7.777 | 78.9 | 3565 | 0.85 |
| bedroom | 7.684 | 7.988 | 74.7 | 6682 | 1.59 |
| kitchen | 3.572 | 3.739 | 87.9 | 4108 | 0.98 |
| bathroom | 1.635 | 1.768 | 83.4 | 6409 | 1.53 |
| hallway | 3.102 | 3.239 | 77.7 | 4976 | 1.18 |
| other_room | 12.809 | 12.919 | 95.1 | 17829 | 4.25 |
| outdoor | 5.272 | 5.413 | 86.8 | 6466 | 1.54 |
| background | 47.383 | 47.474 | 100.0 | 4699 | 1.12 |
| toilet | 0.152 | 0.140 | 96.4 | 5844 | 1.39 |
| sink | 0.342 | 0.325 | 97.9 | 11044 | 2.63 |
| bath_shower | 0.086 | 0.083 | 92.4 | 4991 | 1.19 |
| appliance | 0.663 | 0.652 | 97.0 | 14593 | 3.47 |

## Size constants (`configs/stage2/constraint_constants.yaml`)

Measured on the train split at completion scale (walls 3 px).

| constant | value | how |
|---|---|---|
| T_wall | 3 | length-weighted median wall-polygon thickness = 3.00 px (3 by construction: every plan is scaled so this is 3) |
| T_wall_max | 8 | ceil(length-weighted p98 of wall-polygon thickness = 7.93 px) |
| Lmin_door | 5 | floor(p2 of door_len = 5.90 px), n = 42004 |
| Lmax_door | 27 | ceil(p98 of door_len = 26.32 px) |
| Lmin_win | 4 | floor(p2 of window_len = 4.25 px), n = 36985 |
| Lmax_win | 49 | ceil(p98 of window_len = 48.91 px) |
| k_clear | 1 | kernel 2*floor(d) - 1, d = p2 of door-to-nearest-fixture distance = 1.00 px (so ~2% of real doors violate), n = 40851 doors |
| k_fix | 23 | kernel 2*ceil(r) + 1, r = p98 of the farthest fixture pixel's distance to a room = 11.00 px, n = 32613 fixtures |
| k_living | 23 | floor(p2 of the largest square fitting in each component = 23.0 px), n = 3565 components |
| k_bedroom | 21 | floor(p2 of the largest square fitting in each component = 21.0 px), n = 6682 components |
| k_kitchen | 3 | floor(p2 of the largest square fitting in each component = 3.0 px), n = 4108 components |
| k_bathroom | 1 | floor(p2 of the largest square fitting in each component = 1.0 px), n = 6409 components |
| k_hallway | 7 | floor(p2 of the largest square fitting in each component = 7.0 px), n = 4976 components |
| k_other_room | 3 | floor(p2 of the largest square fitting in each component = 3.0 px), n = 17829 components |
| k_outdoor | 3 | floor(p2 of the largest square fitting in each component = 3.0 px), n = 6466 components |
| k_toilet | 1 | floor(p2 of the largest square fitting in each component = 1.0 px), n = 5844 components |
| k_sink | 3 | floor(p2 of the largest square fitting in each component = 3.0 px), n = 11044 components |
| k_bath_shower | 1 | floor(p2 of the largest square fitting in each component = 1.0 px), n = 4991 components |
| k_appliance | 3 | floor(p2 of the largest square fitting in each component = 3.0 px), n = 14593 components |

**C6, spaces that touch directly in real plans** (direct contact / all adjacency; pairs above 1% are excluded from C6):

| pair | direct-contact rate | excluded |
|---|---|---|
| living-hallway | 0.622 | yes |
| living-kitchen | 0.621 | yes |
| outdoor-background | 0.481 | yes |
| kitchen-hallway | 0.361 | yes |
| living-other_room | 0.263 | yes |
| kitchen-other_room | 0.221 | yes |
| hallway-other_room | 0.095 | yes |
| other_room-background | 0.047 | yes |
| other_room-outdoor | 0.047 | yes |
| living-bedroom | 0.029 | yes |
| bedroom-kitchen | 0.026 | yes |
| bedroom-other_room | 0.019 | yes |
| bathroom-other_room | 0.015 | yes |
| bedroom-hallway | 0.011 | yes |
| hallway-background | 0.007 |  |
| kitchen-background | 0.004 |  |
| living-background | 0.004 |  |
| bathroom-hallway | 0.004 |  |
| kitchen-bathroom | 0.002 |  |
| bathroom-background | 0.002 |  |
| bedroom-background | 0.002 |  |
| hallway-outdoor | 0.002 |  |
| living-bathroom | 0.001 |  |
| bedroom-outdoor | 0.001 |  |
| bedroom-bathroom | 0.001 |  |
| living-outdoor | 0.001 |  |
| kitchen-outdoor | 0.000 |  |
| bathroom-outdoor | 0.000 |  |

**C15, compatible rooms per fixture** (share of the fixture's pixels lying on each room type):

| fixture | living | bedroom | kitchen | bathroom | hallway | other_room | outdoor | background | A(f) |
|---|---|---|---|---|---|---|---|---|---|
| toilet | 0.0 | 0.0 | 0.0 | 79.9 | 0.0 | 20.0 | 0.0 | 0.1 | bathroom, other_room |
| sink | 0.2 | 0.0 | 44.1 | 36.3 | 0.2 | 19.0 | 0.0 | 0.2 | kitchen, bathroom, other_room |
| bath_shower | 0.0 | 0.0 | 0.0 | 77.6 | 0.0 | 22.1 | 0.0 | 0.2 | bathroom, other_room |
| appliance | 0.8 | 0.1 | 63.9 | 11.7 | 1.1 | 22.2 | 0.0 | 0.2 | kitchen, bathroom, other_room |

Fixture outlines overlapping wall/door/window/railing (not painted there): 441901 of 4221528 fixture pixels (10.47%).

## Raw CubiCasa names (all splits)

Fixtures painted: Sink 7891, Toilet 6980, Shower 6049, ElectricalAppliance IntegratedStove 4627, ElectricalAppliance Refrigerator 4609, ElectricalAppliance 3065, ElectricalAppliance SpaceForAppliance2 2574, DoubleSink 2329, ElectricalAppliance WashingMachine 2267, RoundSink 1844, ElectricalAppliance Dishwasher 1809, DoubleSinkRight 952, ElectricalAppliance SpaceForAppliance 841, ElectricalAppliance TumbleDryer 495, Bathtub 395, ElectricalAppliance Heater 255, ShowerCab 245, ElectricalAppliance SaunaStoveRound 147, WoodStove 142, ElectricalAppliance GEARound 111, SideSink 111, ElectricalAppliance IntegratedStoveSmall 77, ElectricalAppliance Fan 66, ElectricalAppliance Stove 63, ElectricalAppliance HighHeater 56, WaterTap 47, Jacuzzi 38, CornerSink 33, ElectricalAppliance DoubleRefrigerator 27, ShowerPlatform 20, Urinal 6, BathtubRound 6, GasStove 5, WashingMachine 1, IntegratedStove 1, Dishwasher 1, GeneralAppliance 1

Fixtures ignored: Closet 31819, BaseCabinet 17325, CoatCloset 12179, WallCabinet 9503, ShowerScreen 4124, SaunaBenchMid 3066, ElectricalAppliance SaunaStove 3020, SaunaBench 3011, Housing 2958, SaunaBenchHigh 2629, SaunaBenchLow 2169, Misc 1773, Fireplace 1236, Chimney 782, FireplaceCorner 480, CoatRack 254, PlaceForFireplace 75, ElectricalAppliance SaunaStove Round 72, FireplaceRound 71, ShowerScreenRoundRight 51, ShowerScreenRoundLeft 49, ClosetTriangle 47, BaseCabinetRound 37, CounterTop 34, PlaceForFireplaceCorner 30, BaseCabinetTriangle 28, ClosetRound 10, PlaceForFireplaceRound 4, SaunaStove 1

Space types: Undefined 11005, Bedroom 7993, Outdoor 7852, Bath 7155, Kitchen 4548, Entry 4211, LivingRoom 3958, Closet 2695, UserDefined 1856, Room 1855, Storage 1809, DraughtLobby 1667, Utility 1015, Dining 954, TechnicalRoom 474, Garage 414, DressingRoom 346, Den 306, CarPort 279, Hall 172, Alcove 164, Office 135, Sauna 133, Elevated 100, Attic 93, RecreationRoom 84, Basement 51, Library 42, SwimmingPool 7, RetailSpace 7, Elevator 5, OpenToBelow 3, Garbage 3, ExerciseRoom 2, Bar 1

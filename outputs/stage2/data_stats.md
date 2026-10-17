# Stage 2 data: CubiCasa5K rasterised from SVG

Built 2026-10-17 by `python -m src.stage2.data.build_cubicasa` on Kaggle (source: Kaggle dataset `qmarva/cubicasa5k`, the complete mirror of Zenodo 2613548). Parser set: greyscale drawing + labels, median wall 8 px, long side <= 2048. Completion set: labels only, median wall 3 px. Official split kept.

## Plans

| split | listed | kept (parser set) | dropped | untyped | completion set |
|---|---|---|---|---|---|
| train | 4200 | 4200 | 0 | 256 | 3944 |
| val | 400 | 400 | 0 | 29 | 371 |
| test | 400 | 400 | 0 | 29 | 371 |
| total | 5000 | 5000 | 0 | 314 | 4686 |

**Untyped plans** (raw space type ['Undefined'] covers > 50% of the room pixels): 314. They stay in the parser set with their room-type pixels written as 255 (ignore), and are left out of the completion set and of every statistic below.

Plan folders with a model.svg: 5000; listed in the official split files: 5000 (0 not in any list, not used).

**Dropped plans and why:**

none

**Drawing images fixed** (parser set): 40 plans had transparent pixels (composited onto white); 16 plans had baked-in black padding turned white (median 5.1% of the image, max 44.3%): high_quality_architectural-585, high_quality_architectural-1386, high_quality_architectural-353, high_quality_architectural-9909, high_quality_architectural-10634, high_quality_architectural-494, high_quality_architectural-5124, high_quality_architectural-20021, high_quality_architectural-1582, high_quality_architectural-1264.

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
| wall | 6.926 | 6.099 | 100.0 | 44280 | 11.23 |
| door | 0.808 | 0.651 | 100.0 | 39094 | 9.91 |
| window | 1.509 | 1.325 | 99.6 | 29721 | 7.54 |
| railing | 0.457 | 0.358 | 74.3 | 7300 | 1.85 |
| living | 7.897 | 8.160 | 83.8 | 3557 | 0.90 |
| bedroom | 8.049 | 8.375 | 79.2 | 6664 | 1.69 |
| kitchen | 3.807 | 3.987 | 92.9 | 4052 | 1.03 |
| bathroom | 1.778 | 1.918 | 87.7 | 6319 | 1.60 |
| hallway | 3.428 | 3.581 | 82.9 | 5073 | 1.29 |
| other_room | 11.024 | 11.050 | 94.8 | 15650 | 3.97 |
| outdoor | 5.783 | 5.933 | 87.8 | 6338 | 1.61 |
| background | 47.552 | 47.624 | 100.0 | 4407 | 1.12 |
| toilet | 0.152 | 0.140 | 96.5 | 5523 | 1.40 |
| sink | 0.343 | 0.325 | 97.9 | 10428 | 2.64 |
| bath_shower | 0.084 | 0.082 | 92.6 | 4702 | 1.19 |
| appliance | 0.401 | 0.391 | 96.0 | 8637 | 2.19 |

## Size constants (`configs/stage2/constraint_constants.yaml`)

Measured on the train split at completion scale (walls 3 px).

| constant | value | how |
|---|---|---|
| T_wall | 3 | length-weighted median wall-polygon thickness = 3.00 px (3 by construction: every plan is scaled so this is 3) |
| T_wall_max | 8 | ceil(length-weighted p98 of wall-polygon thickness = 7.92 px) |
| Lmin_door | 5 | floor(p2 of door_len = 5.98 px), n = 39770 |
| Lmax_door | 27 | ceil(p98 of door_len = 26.49 px) |
| Lmin_win | 4 | floor(p2 of window_len = 4.24 px), n = 34793 |
| Lmax_win | 49 | ceil(p98 of window_len = 48.93 px) |
| k_clear | 7 | kernel 2*floor(d) - 1, d = p10 of door-to-nearest-fixture distance = 4.00 px (so ~10% of real doors violate; C14 is GT-hinged), n = 38690 doors |
| k_fix | 23 | kernel 2*ceil(r) + 1, r = p98 of the farthest fixture pixel's distance to a room = 11.00 px, n = 26946 fixtures |
| k_living | 24 | p2 over pixels of the largest square opening each pixel survives = 24 px (1.81% of pixels survive less), 19341820 px |
| k_bedroom | 25 | p2 over pixels of the largest square opening each pixel survives = 25 px (1.81% of pixels survive less), 19714258 px |
| k_kitchen | 7 | p2 over pixels of the largest square opening each pixel survives = 7 px (1.51% of pixels survive less), 9324245 px |
| k_bathroom | 3 | p2 over pixels of the largest square opening each pixel survives = 3 px (1.38% of pixels survive less), 4355052 px |
| k_hallway | 10 | p2 over pixels of the largest square opening each pixel survives = 10 px (1.63% of pixels survive less), 8395073 px |
| k_other_room | 11 | p2 over pixels of the largest square opening each pixel survives = 11 px (1.69% of pixels survive less), 27001053 px |
| k_outdoor | 13 | p2 over pixels of the largest square opening each pixel survives = 13 px (1.61% of pixels survive less), 14165050 px |
| k_toilet | 3 | p2 over pixels of the largest square opening each pixel survives = 3 px (1.51% of pixels survive less), 371934 px |
| k_sink | 4 | p2 over pixels of the largest square opening each pixel survives = 4 px (0.68% of pixels survive less), 839669 px |
| k_bath_shower | 1 | p2 over pixels of the largest square opening each pixel survives = 1 px (0.00% of pixels survive less), 206689 px |
| k_appliance | 5 | p2 over pixels of the largest square opening each pixel survives = 5 px (0.64% of pixels survive less), 982251 px |

**C6, spaces that touch directly in real plans** (direct contact / all adjacency, sorted; pairs at or above 10% are excluded from C6, never a pair with ['bathroom'], always [['outdoor', 'background']]):

| pair | direct-contact rate | excluded from C6 |
|---|---|---|
| living-hallway | 0.620 | yes |
| living-kitchen | 0.609 | yes |
| outdoor-background | 0.494 | yes |
| kitchen-hallway | 0.344 | yes |
| living-other_room | 0.254 | yes |
| kitchen-other_room | 0.210 | yes |
| hallway-other_room | 0.086 |  |
| living-bedroom | 0.028 |  |
| other_room-outdoor | 0.025 |  |
| bedroom-kitchen | 0.024 |  |
| bedroom-other_room | 0.019 |  |
| other_room-background | 0.018 |  |
| bathroom-other_room | 0.015 |  |
| bedroom-hallway | 0.011 |  |
| hallway-background | 0.007 |  |
| kitchen-background | 0.004 |  |
| living-background | 0.004 |  |
| bathroom-hallway | 0.004 |  |
| bathroom-background | 0.003 |  |
| kitchen-bathroom | 0.003 |  |
| hallway-outdoor | 0.002 |  |
| bedroom-background | 0.002 |  |
| living-bathroom | 0.001 |  |
| bedroom-outdoor | 0.001 |  |
| bedroom-bathroom | 0.001 |  |
| living-outdoor | 0.001 |  |
| kitchen-outdoor | 0.000 |  |
| bathroom-outdoor | 0.000 |  |

**R1, room area range per type** (px at completion scale):

| room | min | max | how |
|---|---|---|---|
| living | 560 | 20417 | p1 / p99 of 4-connected room areas (px), n = 3555 rooms |
| bedroom | 578 | 10296 | p1 / p99 of 4-connected room areas (px), n = 6664 rooms |
| kitchen | 256 | 11393 | p1 / p99 of 4-connected room areas (px), n = 3826 rooms |
| bathroom | 99 | 3420 | p1 / p99 of 4-connected room areas (px), n = 6051 rooms |
| hallway | 80 | 8181 | p1 / p99 of 4-connected room areas (px), n = 5072 rooms |
| other_room | 23 | 10898 | p1 / p99 of 4-connected room areas (px), n = 15581 rooms |

**R3, door-connected space pairs** (train; each door's two faces; sorted by count):

| pair | doors | % of door connections | R3 |
|---|---|---|---|
| bathroom-other_room | 4357 | 11.17 |  |
| other_room-other_room | 3568 | 9.15 |  |
| bedroom-hallway | 3514 | 9.01 |  |
| hallway-other_room | 3135 | 8.04 |  |
| bathroom-hallway | 3132 | 8.03 |  |
| bedroom-other_room | 2672 | 6.85 |  |
| other_room-outdoor | 2467 | 6.33 |  |
| living-outdoor | 2142 | 5.49 |  |
| hallway-outdoor | 1826 | 4.68 |  |
| other_room-background | 1774 | 4.55 |  |
| hallway-background | 1530 | 3.92 |  |
| living-bedroom | 1408 | 3.61 |  |
| hallway-hallway | 1257 | 3.22 |  |
| kitchen-other_room | 1005 | 2.58 |  |
| kitchen-hallway | 810 | 2.08 |  |
| living-other_room | 768 | 1.97 |  |
| living-kitchen | 426 | 1.09 |  |
| living-hallway | 401 | 1.03 |  |
| bedroom-outdoor | 388 | 1.00 |  |
| bedroom-kitchen | 357 | 0.92 |  |
| kitchen-outdoor | 270 | 0.69 |  |
| living-bathroom | 245 | 0.63 |  |
| living-background | 235 | 0.60 |  |
| bathroom-bathroom | 220 | 0.56 |  |
| kitchen-bathroom | 213 | 0.55 |  |
| bedroom-bathroom | 135 | 0.35 | rare, allowed (user) |
| outdoor-outdoor | 117 | 0.30 | forbidden |
| outdoor-background | 111 | 0.29 | forbidden |
| living-living | 92 | 0.24 | forbidden |
| bathroom-outdoor | 92 | 0.24 | rare, allowed (user) |
| kitchen-background | 69 | 0.18 | forbidden |
| background-background | 65 | 0.17 | forbidden |
| bedroom-bedroom | 64 | 0.16 | forbidden |
| bedroom-background | 62 | 0.16 | forbidden |
| kitchen-kitchen | 45 | 0.11 | forbidden |
| bathroom-background | 28 | 0.07 | forbidden (user) |

space-type pairs on < 0.5% of 39000 door connections (train); pairs with a bathroom are listed separately for review

**Appliance sub-types** (train, typed plans; room type each instance stands in; appliance if >= 50% in kitchens):

| sub-type | instances | % in kitchen | decision | bathroom | bedroom | hallway | kitchen | living | none | other_room |
|---|---|---|---|---|---|---|---|---|---|---|
| IntegratedStove | 3669 | 93.5 | appliance | 0 | 2 | 13 | 3429 | 17 | 9 | 199 |
| Refrigerator | 3653 | 92.2 | appliance | 0 | 0 | 28 | 3367 | 38 | 8 | 212 |
| ElectricalAppliance | 2453 | 43.1 | ignored | 141 | 4 | 235 | 1058 | 17 | 11 | 987 |
| SpaceForAppliance2 | 2191 | 44.7 | ignored | 766 | 0 | 9 | 980 | 7 | 5 | 424 |
| WashingMachine | 1764 | 1.6 | ignored | 895 | 0 | 0 | 29 | 0 | 3 | 837 |
| Dishwasher | 1367 | 91.0 | appliance | 5 | 1 | 3 | 1244 | 8 | 3 | 103 |
| SpaceForAppliance | 699 | 58.8 | appliance | 142 | 0 | 5 | 411 | 9 | 0 | 132 |
| TumbleDryer | 383 | 0.8 | ignored | 134 | 0 | 0 | 3 | 0 | 1 | 245 |
| Heater | 219 | 8.7 | ignored | 6 | 4 | 0 | 19 | 7 | 1 | 182 |
| WoodStove | 119 | 81.5 | appliance | 0 | 1 | 0 | 97 | 4 | 0 | 17 |
| GEARound | 94 | 3.2 | ignored | 11 | 0 | 4 | 3 | 1 | 1 | 74 |
| IntegratedStoveSmall | 61 | 80.3 | appliance | 0 | 0 | 2 | 49 | 1 | 0 | 9 |
| Fan | 55 | 87.3 | appliance | 0 | 0 | 0 | 48 | 0 | 0 | 7 |
| Stove | 45 | 91.1 | appliance | 0 | 0 | 0 | 41 | 0 | 0 | 4 |
| HighHeater | 44 | 20.5 | ignored | 0 | 2 | 0 | 9 | 3 | 0 | 30 |
| DoubleRefrigerator | 22 | 95.5 | appliance | 0 | 0 | 0 | 21 | 1 | 0 | 0 |
| GasStove | 2 | 100.0 | appliance | 0 | 0 | 0 | 2 | 0 | 0 | 0 |
| GeneralAppliance | 1 | 100.0 | appliance | 0 | 0 | 0 | 1 | 0 | 0 | 0 |

**C15, compatible rooms per fixture** (share of the fixture's pixels lying on each room type):

| fixture | living | bedroom | kitchen | bathroom | hallway | other_room | outdoor | background | A(f) |
|---|---|---|---|---|---|---|---|---|---|
| toilet | 0.0 | 0.0 | 0.0 | 83.3 | 0.0 | 16.5 | 0.0 | 0.1 | bathroom, other_room |
| sink | 0.2 | 0.0 | 46.3 | 37.8 | 0.2 | 15.2 | 0.0 | 0.2 | kitchen, bathroom, other_room |
| bath_shower | 0.0 | 0.0 | 0.0 | 81.9 | 0.0 | 17.8 | 0.0 | 0.3 | bathroom, other_room |
| appliance | 1.0 | 0.0 | 90.2 | 1.5 | 0.4 | 6.7 | 0.0 | 0.2 | kitchen, other_room |

Fixture outlines overlapping wall/door/window/railing (not painted there): 315087 of 3155286 fixture pixels (9.99%).

## Raw CubiCasa names (all splits)

Fixtures painted: Sink 7412, Toilet 6574, Shower 5689, ElectricalAppliance IntegratedStove 4353, ElectricalAppliance Refrigerator 4350, DoubleSink 2196, RoundSink 1735, ElectricalAppliance Dishwasher 1701, DoubleSinkRight 905, ElectricalAppliance SpaceForAppliance 808, Bathtub 374, ShowerCab 227, WoodStove 136, SideSink 106, ElectricalAppliance IntegratedStoveSmall 71, ElectricalAppliance Fan 65, ElectricalAppliance Stove 60, WaterTap 43, Jacuzzi 36, CornerSink 33, ElectricalAppliance DoubleRefrigerator 26, ShowerPlatform 20, Urinal 6, BathtubRound 6, GasStove 5, IntegratedStove 1, Dishwasher 1, GeneralAppliance 1

Fixtures ignored: Closet 30010, BaseCabinet 16335, CoatCloset 11731, WallCabinet 9055, ShowerScreen 3761, ElectricalAppliance 2935, SaunaBenchMid 2911, ElectricalAppliance SaunaStove 2892, SaunaBench 2885, Housing 2854, SaunaBenchHigh 2494, ElectricalAppliance SpaceForAppliance2 2455, ElectricalAppliance WashingMachine 2153, SaunaBenchLow 2067, Misc 1675, Fireplace 1173, Chimney 720, ElectricalAppliance TumbleDryer 464, FireplaceCorner 452, ElectricalAppliance Heater 244, CoatRack 243, ElectricalAppliance SaunaStoveRound 123, ElectricalAppliance GEARound 106, PlaceForFireplace 74, ElectricalAppliance SaunaStove Round 70, FireplaceRound 64, ElectricalAppliance HighHeater 56, ShowerScreenRoundRight 46, ClosetTriangle 44, ShowerScreenRoundLeft 43, BaseCabinetRound 35, CounterTop 32, PlaceForFireplaceCorner 30, BaseCabinetTriangle 26, ClosetRound 10, PlaceForFireplaceRound 4, SaunaStove 1, WashingMachine 1

Space types: Undefined 8496, Bedroom 7963, Outdoor 7476, Bath 7090, Kitchen 4516, Entry 4189, LivingRoom 3950, Closet 2676, Room 1842, UserDefined 1826, Storage 1796, DraughtLobby 1661, Utility 1013, Dining 953, TechnicalRoom 468, Garage 412, DressingRoom 341, Den 302, CarPort 278, Hall 172, Alcove 162, Office 133, Sauna 127, Elevated 98, Attic 90, RecreationRoom 84, Basement 51, Library 41, SwimmingPool 7, RetailSpace 7, OpenToBelow 3, Garbage 3, Elevator 3, ExerciseRoom 2, Bar 1

"""Pipeline messages list only real additions (no space classes, no duplicates), with the 'had no door'
wording; the type-restricted fill keeps only the detected type."""

import numpy as np

from src.stage2.constraints import toys
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import describe_additions, restrict

SPACE_NAMES = set(C16.cfg["groups"]["S"]) | {"outdoor", "background"}


def added_part(m):
    """The 'X added' part of a message (what was added), without the location."""
    return m.split("->")[-1].split(" added")[0]


def test_door_message_and_no_space_classes():
    after = toys.valid_plan()
    before = toys.broken("C8")[0]                    # bathroom door walled up; the fill put it back
    holes = [{"type": "door", "rules": ["C8"], "box": (44, 52, 50, 65)}]
    msgs = describe_additions(before, after, holes)
    assert any("had no door -> door added" in m for m in msgs)
    for m in msgs:
        assert not any(n in added_part(m).split() for n in SPACE_NAMES)


def test_fixture_message_one_line_each():
    before, after = toys.broken("C17")[0], toys.valid_plan()
    holes = [{"type": "toilet", "rules": ["detector"], "box": (55, 50, 80, 69)}]
    msgs = describe_additions(before, after, holes)
    assert any(m.startswith("toilet added in") for m in msgs)
    assert any(m.startswith("bath_shower added in") for m in msgs)
    assert len(msgs) == len(set(msgs))


def test_no_duplicate_lines():
    after = toys.valid_plan()
    before = toys.broken("C8")[0]
    holes = [{"type": "door", "rules": ["C8"], "box": (44, 52, 50, 65)}] * 2
    msgs = describe_additions(before, after, holes)
    assert len(msgs) == len(set(msgs))


def test_restrict_keeps_only_detected_type():
    before, after = toys.broken("C17")[0], toys.valid_plan()      # the fill added toilet + bath_shower
    kept = restrict(before, after, "toilet")
    changed = kept != before
    assert changed.any() and (kept[changed] == C16.index["toilet"]).all()
    assert np.array_equal(restrict(before, after, "window"), before)

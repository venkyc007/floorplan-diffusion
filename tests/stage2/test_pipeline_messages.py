"""Pipeline messages list only real additions (no space classes), with the 'had no door' wording."""

from src.stage2.constraints import toys
from src.stage2.data.classes16 import C16
from src.stage2.pipeline.run import describe_additions


def test_door_message_and_no_space_classes():
    after = toys.valid_plan()
    before = toys.broken("C8")[0]                    # bathroom door walled up; the fill put it back
    holes = [{"type": "door", "rules": ["C8"], "box": (44, 52, 50, 65)}]
    msgs = describe_additions(before, after, holes)
    assert any("had no door -> door added" in m for m in msgs)
    names = set(C16.cfg["groups"]["S"]) | {"outdoor", "background"}
    for m in msgs:
        added = m.split("->")[1]
        assert not any(f" {n}" in added.split(" added")[0] for n in names)


def test_fixture_message():
    before, after = toys.broken("C17")[0], toys.valid_plan()
    holes = [{"type": "toilet", "rules": ["detector"], "box": (55, 50, 80, 69)}]
    msgs = describe_additions(before, after, holes)
    assert "toilet" in msgs[0] and "bath_shower" in msgs[0] and "bathroom" not in msgs[0].split("->")[1].split(" added")[0]

"""CHECK flag merging: the merged type follows the evidence, not the raw score (audit 2026-10-21, CHK-1 / CHK-2)."""

from src.stage2.detect.rules import Candidate, merge


def test_detector_room_beats_an_overlapping_rule_door():
    out = merge([Candidate("room", (10, 10, 50, 50), ["detector"], 0.97, (30, 30)),
                 Candidate("door", (40, 40, 48, 60), ["R5"], 14.0)])
    assert len(out) == 1 and out[0].type == "room"
    assert out[0].box == (10, 10, 50, 60) and out[0].rules == ["R5", "detector"]
    assert out[0].peak == (30, 30)


def test_region_flag_always_wins_and_keeps_the_union_box():
    out = merge([Candidate("region", (10, 10, 90, 90), ["R8"], 5.0),
                 Candidate("door", (40, 85, 48, 95), ["C8"], 900.0)])
    assert [(c.type, c.box) for c in out] == [("region", (10, 10, 90, 95))]


def test_rule_type_beats_a_detector_type_other_than_room():
    out = merge([Candidate("wall", (0, 0, 10, 10), ["detector"], 0.8),
                 Candidate("door", (5, 5, 12, 12), ["R5"], 14.0)])
    assert out[0].type == "door"


def test_separate_boxes_stay_separate():
    out = merge([Candidate("door", (0, 0, 5, 5), ["R5"], 12.0), Candidate("window", (20, 20, 25, 25), ["detector"], 0.7)])
    assert sorted(c.type for c in out) == ["door", "window"]

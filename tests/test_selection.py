"""Unit tests for the Pareto front and knee in src/selection.py. Run:  python -m pytest tests/"""

import pandas as pd

from src.selection import X, Y, knee, pareto_front


def pts(rows):
    return pd.DataFrame(rows, columns=["name", X, Y]).set_index("name")


def test_pareto_front_marks_dominated_points():
    p = pts([("a", 0.70, 90), ("b", 0.80, 85), ("c", 0.85, 70), ("d", 0.75, 80), ("e", 0.80, 85)])
    front = dict(zip(p.index, pareto_front(p)))
    assert front == {"a": True, "b": True, "c": True, "d": False, "e": True}   # ties are not dominated


def test_knee_is_the_bend_of_the_front():
    # an L-shaped front: the corner point "bend" is far from the chord a-c
    p = pts([("a", 0.70, 90), ("bend", 0.84, 88), ("c", 0.85, 60), ("x", 0.72, 70)])
    front = pareto_front(p)
    assert knee(p, front)[0] == "bend"


def test_knee_with_two_front_points_uses_largest_sum():
    p = pts([("a", 0.70, 90), ("b", 0.85, 89), ("x", 0.60, 50)])
    assert knee(p, pareto_front(p))[0] == "b"

from collections import deque

from src.tracker import estimate_speed


def test_estimate_speed_uses_last_two_points():
    trails = {1: deque([(0, 0), (10, 0)], maxlen=10)}
    speeds = estimate_speed(trails, fps=30, px_per_meter=10)
    assert speeds[1] == 108.0


def test_estimate_speed_short_trail_is_zero():
    trails = {1: deque([(5, 5)], maxlen=10)}
    assert estimate_speed(trails, fps=30)[1] == 0.0


def test_estimate_speed_multiple_tracks():
    trails = {
        1: deque([(0, 0), (0, 10)], maxlen=10),
        2: deque([(3, 4)], maxlen=10),
    }
    speeds = estimate_speed(trails, fps=10, px_per_meter=5)
    assert speeds[1] == 72.0
    assert speeds[2] == 0.0

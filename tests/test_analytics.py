import pandas as pd

from src.analytics import (
    _classify_density,
    _find_congestion_windows,
    build_dataframe,
    compute_insights,
)


def _logs():
    return [
        {
            "frame_id": 1,
            "timestamp_s": 0,
            "counts": {"car": 2, "person": 1},
            "total_in_frame": 3,
            "fps": 30,
            "inference_ms": 20,
        },
        {
            "frame_id": 2,
            "timestamp_s": 1,
            "counts": {"car": 20},
            "total_in_frame": 20,
            "fps": 30,
            "inference_ms": 25,
        },
        {
            "frame_id": 3,
            "timestamp_s": 2,
            "counts": {"car": 1},
            "total_in_frame": 1,
            "fps": 30,
            "inference_ms": 18,
        },
    ]


def test_classify_density_boundaries():
    assert _classify_density(0) == "free"
    assert _classify_density(3) == "light"
    assert _classify_density(8) == "moderate"
    assert _classify_density(15) == "heavy"
    assert _classify_density(25) == "congested"


def test_build_dataframe_schema():
    df = build_dataframe(_logs())
    assert list(df["frame_id"]) == [1, 2, 3]
    assert "car" in df.columns
    assert "person" in df.columns
    assert list(df["density_label"]) == ["light", "heavy", "free"]
    assert list(df["time_str"]) == ["0:00:00", "0:00:01", "0:00:02"]


def test_build_dataframe_empty_counts():
    df = build_dataframe([{"frame_id": 1, "timestamp_s": 0, "counts": {}}])
    assert df.loc[0, "total"] == 0
    assert df.loc[0, "density_label"] == "free"


def test_compute_insights_peak_and_distribution():
    insights = compute_insights(build_dataframe(_logs()))
    assert insights["total_frames"] == 3
    assert insights["peak_frame"] == 2
    assert insights["max_vehicles_in_frame"] == 20
    assert insights["density_distribution_%"]["heavy"] == 33.3
    assert insights["class_totals"]["car"] == 23


def test_compute_insights_empty_dataframe():
    assert compute_insights(pd.DataFrame()) == {}


def test_find_congestion_windows():
    windows = _find_congestion_windows(build_dataframe(_logs()))
    assert len(windows) == 1
    assert windows[0]["start_s"] == 1
    assert windows[0]["end_s"] == 2

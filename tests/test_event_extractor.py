from __future__ import annotations

import pandas as pd
import pytest

from turf.event_extractor import EventExtractor
from turf.match_loader import MatchData


@pytest.fixture()
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Team": ["Home", "Home", "Away", "Home"],
            "Type": ["pass", "other", "pass", "pass"],
            "Subtype": ["success", None, "fail", "success"],
            "Period": [1, 1, 1, 2],
            "Start Frame": [2, 5, 10, 20],
            "Start Time [s]": [0.06, 0.15, 0.30, 0.60],
            "End Frame": [4, 9, 14, 24],
            "End Time [s]": [0.12, 0.28, 0.44, 0.76],
            "From": ["Alice", "Bob", "Charlie", "Alice"],
            "To": ["Bob", None, "Dave", "Charlie"],
            "Start X": [0.1, -5.0, 10.0, 3.0],
            "Start Y": [0.2, -3.0, 5.0, 1.0],
            "End X": [1.0, -4.0, 11.0, 4.0],
            "End Y": [1.2, -2.5, 5.5, 2.0],
        }
    )


@pytest.fixture()
def home_tracking() -> pd.DataFrame:
    n = 30
    return pd.DataFrame(
        {
            "Period": [1] * n,
            "Time [s]": [i * 0.04 for i in range(n)],
            "Home_1_x": [float(i) for i in range(n)],
            "Home_1_y": [float(i) * 0.5 for i in range(n)],
            "ball_x": [float(i) * 0.1 for i in range(n)],
            "ball_y": [float(i) * 0.1 for i in range(n)],
        }
    )


@pytest.fixture()
def away_tracking() -> pd.DataFrame:
    n = 30
    return pd.DataFrame(
        {
            "Period": [1] * n,
            "Time [s]": [i * 0.04 for i in range(n)],
            "Away_1_x": [float(-i) for i in range(n)],
            "Away_1_y": [float(-i) * 0.5 for i in range(n)],
            "ball_x": [float(i) * 0.1 for i in range(n)],
            "ball_y": [float(i) * 0.1 for i in range(n)],
        }
    )


@pytest.fixture()
def match_data(
    events: pd.DataFrame,
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
) -> MatchData:
    return MatchData(
        match_id="10502",
        events=events,
        home_tracking=home_tracking,
        away_tracking=away_tracking,
    )


# ---------------------------------------------------------------------------
# EventExtractor.list_labels
# ---------------------------------------------------------------------------


def test_list_labels_returns_sorted_unique(events: pd.DataFrame) -> None:
    labels = EventExtractor().list_labels(events)
    assert labels == ["other", "pass"]


def test_list_labels_case_normalized() -> None:
    df = pd.DataFrame({"Type": ["PASS", "OTHER", "Pass", "other"]})
    labels = EventExtractor().list_labels(df)
    assert labels == ["other", "pass"]


def test_list_labels_empty_events_returns_empty_list() -> None:
    df = pd.DataFrame({"Type": pd.Series([], dtype=str)})
    labels = EventExtractor().list_labels(df)
    assert labels == []


# ---------------------------------------------------------------------------
# EventExtractor.extract — filtering
# ---------------------------------------------------------------------------


def test_extract_returns_correct_clip_count(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    assert len(clips) == 3


def test_extract_case_insensitive(match_data: MatchData) -> None:
    assert len(EventExtractor().extract(match_data, "PASS")) == 3
    assert len(EventExtractor().extract(match_data, "Pass")) == 3


def test_extract_unknown_label_returns_empty_list(match_data: MatchData) -> None:
    assert EventExtractor().extract(match_data, "tackle") == []


# ---------------------------------------------------------------------------
# EventClip structure
# ---------------------------------------------------------------------------


def test_extract_clip_event_idx_sequential(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    assert [c.event_idx for c in clips] == [0, 1, 2]


def test_extract_clip_frame_bounds(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    assert [c.start_frame for c in clips] == [2, 10, 20]
    assert [c.end_frame for c in clips] == [4, 14, 24]


def test_extract_frames_sliced_correctly(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass")

    # clip 0: frames 2–4 inclusive → 3 rows; row 0 is tracking row 2
    assert len(clips[0].home_frames) == 3
    assert len(clips[0].away_frames) == 3
    assert clips[0].home_frames.iloc[0]["Home_1_x"] == 2.0
    assert clips[0].away_frames.iloc[0]["Away_1_x"] == -2.0

    # clip 1: frames 10–14 inclusive → 5 rows
    assert len(clips[1].home_frames) == 5
    assert len(clips[1].away_frames) == 5


def test_extract_metadata_fields(match_data: MatchData) -> None:
    meta = EventExtractor().extract(match_data, "pass")[0].metadata
    assert meta["start_frame"] == 2
    assert meta["end_frame"] == 4
    assert meta["start_x"] == pytest.approx(0.1)
    assert meta["start_y"] == pytest.approx(0.2)
    assert meta["end_x"] == pytest.approx(1.0)
    assert meta["end_y"] == pytest.approx(1.2)
    assert meta["from_player"] == "Alice"
    assert meta["to_player"] == "Bob"
    assert meta["team"] == "Home"
    assert meta["subtype"] == "success"
    assert meta["period"] == 1


# ---------------------------------------------------------------------------
# EventClip — edge case: single-frame event (start == end)
# ---------------------------------------------------------------------------


def test_extract_single_frame_event_has_one_row() -> None:
    df = pd.DataFrame(
        {
            "Team": ["Home"],
            "Type": ["pass"],
            "Subtype": ["success"],
            "Period": [1],
            "Start Frame": [5],
            "Start Time [s]": [0.15],
            "End Frame": [5],
            "End Time [s]": [0.15],
            "From": ["Alice"],
            "To": ["Bob"],
            "Start X": [0.0],
            "Start Y": [0.0],
            "End X": [0.0],
            "End Y": [0.0],
        }
    )
    tracking = pd.DataFrame({"Period": range(10), "Home_1_x": range(10)})
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=tracking,
        away_tracking=tracking.rename(columns={"Home_1_x": "Away_1_x"}),
    )
    clips = EventExtractor().extract(data, "pass")
    assert len(clips[0].home_frames) == 1

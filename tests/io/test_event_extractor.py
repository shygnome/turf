from __future__ import annotations

import warnings

import pandas as pd
import pytest

from turf.io.event_extractor import EventExtractor
from turf.io.match_loader import MatchData


@pytest.fixture()
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Team": ["Home", "Home", "Away", "Home"],
            "Type": ["pass", "other", "pass", "pass"],
            "Subtype": ["success", None, "fail", "success"],
            "Period": [1, 1, 1, 1],
            "Start Frame": [2, 5, 10, 20],
            "Start Time [s]": [2.0, 5.0, 10.0, 20.0],
            "End Frame": [4, 9, 14, 24],
            "End Time [s]": [4.0, 9.0, 14.0, 24.0],
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
            "Time [s]": [float(i) for i in range(n)],
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
            "Time [s]": [float(i) for i in range(n)],
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
    clips = EventExtractor().extract(match_data, "pass", infer_endpoints=False)
    assert [c.start_frame for c in clips] == [2, 10, 20]
    assert [c.end_frame for c in clips] == [4, 14, 24]


def test_extract_frames_sliced_correctly(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass", infer_endpoints=False)

    # clip 0: frames 2–4 inclusive → 3 rows; row 0 is tracking row 2
    assert len(clips[0].home_frames) == 3
    assert len(clips[0].away_frames) == 3
    assert clips[0].home_frames.iloc[0]["Home_1_x"] == 2.0
    assert clips[0].away_frames.iloc[0]["Away_1_x"] == -2.0

    # clip 1: frames 10–14 inclusive → 5 rows
    assert len(clips[1].home_frames) == 5
    assert len(clips[1].away_frames) == 5


def test_extract_metadata_fields(match_data: MatchData) -> None:
    meta = (
        EventExtractor().extract(match_data, "pass", infer_endpoints=False)[0].metadata
    )
    assert meta["start_frame"] == 2
    assert meta["end_frame"] == 4
    assert meta["start_time"] == pytest.approx(2.0)
    assert meta["end_time"] == pytest.approx(4.0)
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
# EventExtractor.extract — frame bounds validation
# ---------------------------------------------------------------------------


def test_extract_warns_and_skips_events_with_no_tracking_for_period() -> None:
    df = pd.DataFrame(
        {
            "Team": ["Home"],
            "Type": ["pass"],
            "Subtype": ["success"],
            "Period": [1],
            "Start Frame": [50],
            "Start Time [s]": [50.0],
            "End Frame": [60],
            "End Time [s]": [60.0],
            "From": ["Alice"],
            "To": ["Bob"],
            "Start X": [0.0],
            "Start Y": [0.0],
            "End X": [1.0],
            "End Y": [1.0],
        }
    )
    # Tracking only has period 2; event is period 1 → skip with warning
    tracking = pd.DataFrame(
        {
            "Period": [2] * 10,
            "Time [s]": [float(i) for i in range(10)],
            "Home_1_x": range(10),
        }
    )
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=tracking,
        away_tracking=tracking.rename(columns={"Home_1_x": "Away_1_x"}),
    )
    with pytest.warns(UserWarning, match="Periods \\[1\\]"):
        clips = EventExtractor().extract(data, "pass")
    assert clips == []


# ---------------------------------------------------------------------------
# EventClip — edge case: single-frame event (start == end)
# ---------------------------------------------------------------------------


def test_extract_frames_exclude_wrong_period_rows() -> None:
    """When tracking rows from multiple periods are interleaved, the extracted
    clip must contain only rows from the event's period, not neighbours."""
    df = pd.DataFrame(
        {
            "Team": ["Home"],
            "Type": ["pass"],
            "Subtype": ["success"],
            "Period": [1],
            "Start Frame": [0],
            "Start Time [s]": [0.0],
            "End Frame": [2],
            "End Time [s]": [2.0],
            "From": ["Alice"],
            "To": ["Bob"],
            "Start X": [0.0],
            "Start Y": [0.0],
            "End X": [1.0],
            "End Y": [1.0],
        }
    )
    # Interleaved periods: P1 rows at labels 0, 2, 4; P2 rows at labels 1, 3
    tracking = pd.DataFrame(
        {
            "Period": [1, 2, 1, 2, 1],
            "Time [s]": [0.0, 0.0, 1.0, 1.0, 2.0],
            "Home_1_x": [1.0, 99.0, 2.0, 99.0, 3.0],
        }
    )
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=tracking,
        away_tracking=tracking.rename(columns={"Home_1_x": "Away_1_x"}),
    )
    clips = EventExtractor().extract(data, "pass")
    assert len(clips) == 1
    assert (clips[0].home_frames["Period"] == 1).all()
    assert 99.0 not in clips[0].home_frames["Home_1_x"].tolist()


def test_extract_clips_valid_when_end_snaps_before_start() -> None:
    """Non-monotonic tracking timestamps can cause end_label < start_label via idxmin().
    The extractor must clamp end_label to start_label so the .loc slice is non-empty."""
    df = pd.DataFrame(
        {
            "Team": ["Home"],
            "Type": ["pass"],
            "Subtype": ["success"],
            "Period": [1],
            "Start Frame": [5],
            "Start Time [s]": [3.0],  # nearest row: label=1 (Time=3.0)
            "End Frame": [10],
            "End Time [s]": [5.0],  # nearest row: label=0 (Time=5.0) → end < start
            "From": ["Alice"],
            "To": ["Bob"],
            "Start X": [0.0],
            "Start Y": [0.0],
            "End X": [1.0],
            "End Y": [1.0],
        }
    )
    # Non-monotonic timestamps: label 0→Time=5, label 1→Time=3, label 2→Time=7
    tracking = pd.DataFrame(
        {
            "Period": [1, 1, 1],
            "Time [s]": [5.0, 3.0, 7.0],
            "Home_1_x": [10.0, 20.0, 30.0],
        }
    )
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=tracking,
        away_tracking=tracking.rename(columns={"Home_1_x": "Away_1_x"}),
    )
    clips = EventExtractor().extract(data, "pass")
    assert len(clips) == 1
    assert len(clips[0].home_frames) >= 1


def test_extract_single_frame_event_has_one_row() -> None:
    df = pd.DataFrame(
        {
            "Team": ["Home"],
            "Type": ["pass"],
            "Subtype": ["success"],
            "Period": [1],
            "Start Frame": [5],
            "Start Time [s]": [5.0],
            "End Frame": [5],
            "End Time [s]": [5.0],
            "From": ["Alice"],
            "To": ["Bob"],
            "Start X": [0.0],
            "Start Y": [0.0],
            "End X": [0.0],
            "End Y": [0.0],
        }
    )
    tracking = pd.DataFrame(
        {
            "Period": [1] * 10,
            "Time [s]": [float(i) for i in range(10)],
            "Home_1_x": range(10),
        }
    )
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=tracking,
        away_tracking=tracking.rename(columns={"Home_1_x": "Away_1_x"}),
    )
    clips = EventExtractor().extract(data, "pass")
    assert len(clips[0].home_frames) == 1


# ---------------------------------------------------------------------------
# EventExtractor.extract — endpoint inference (pass / cross)
# ---------------------------------------------------------------------------


def test_extract_pass_inferred_end_coords_stored_separately(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    # clip 0: next event is 'other' at t=5.0 (Start X=-5.0, Start Y=-3.0)
    # original PFF end_x/y unchanged; inferred values in dedicated columns
    assert clips[0].metadata["end_x"] == pytest.approx(1.0)
    assert clips[0].metadata["end_y"] == pytest.approx(1.2)
    assert clips[0].metadata["inferred_end_x"] == pytest.approx(-5.0)
    assert clips[0].metadata["inferred_end_y"] == pytest.approx(-3.0)


def test_extract_pass_inferred_end_time_stored_separately(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    # original PFF end_time unchanged; inferred values in dedicated columns
    assert clips[0].metadata["end_time"] == pytest.approx(4.0)
    assert clips[0].metadata["inferred_end_time"] == pytest.approx(5.0)


def test_extract_pass_end_frame_reflects_inferred_clip_boundary(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    # clip frame slice uses inferred end_time → end_frame = tracking snap of t=5.0 = 5
    assert clips[0].end_frame == 5


def test_extract_pass_inferred_frames_extended(match_data: MatchData) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    # clip 0: start t=2 → inferred end t=5 → tracking rows 2,3,4,5 = 4 rows
    assert len(clips[0].home_frames) == 4
    assert len(clips[0].away_frames) == 4


def test_extract_pass_last_in_period_has_null_inferred_columns(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "pass")
    # clip 2 (t=20.0): no successor — inferred columns are None
    assert clips[2].metadata["inferred_end_x"] is None
    assert clips[2].metadata["inferred_end_y"] is None
    assert clips[2].metadata["inferred_end_time"] is None


def test_extract_infer_endpoints_false_has_null_inferred_columns(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "pass", infer_endpoints=False)
    assert clips[0].metadata["end_x"] == pytest.approx(1.0)
    assert clips[0].metadata["inferred_end_x"] is None
    assert clips[0].metadata["inferred_end_time"] is None


def test_extract_non_pass_label_has_null_inferred_columns(
    match_data: MatchData,
) -> None:
    clips = EventExtractor().extract(match_data, "other")
    assert clips[0].metadata["end_x"] == pytest.approx(-4.0)
    assert clips[0].metadata["inferred_end_x"] is None


# ---------------------------------------------------------------------------
# EventExtractor.extract — per-period tracking coverage
# ---------------------------------------------------------------------------


def _make_tracking_with_periods(periods: list[int]) -> pd.DataFrame:
    rows = []
    t = 0.0
    for p in periods:
        for _ in range(5):
            rows.append({"Period": p, "Time [s]": t, "Home_1_x": t})
            t += 1.0
    return pd.DataFrame(rows)


def test_extract_skips_uncovered_period_keeps_covered_events() -> None:
    """Events in periods 1 and 3; tracking covers only period 1.
    Period-3 events are skipped; period-1 clips are returned."""
    tracking = _make_tracking_with_periods([1])
    away = tracking.rename(columns={"Home_1_x": "Away_1_x"})
    df = pd.DataFrame(
        {
            "Team": ["Home", "Home", "Away", "Away"],
            "Type": ["pass", "pass", "pass", "pass"],
            "Subtype": ["success"] * 4,
            "Period": [1, 1, 3, 3],
            "Start Frame": [0, 1, 0, 1],
            "Start Time [s]": [0.0, 1.0, 0.0, 1.0],
            "End Frame": [1, 2, 1, 2],
            "End Time [s]": [1.0, 2.0, 1.0, 2.0],
            "From": ["Alice"] * 4,
            "To": ["Bob"] * 4,
            "Start X": [0.0] * 4,
            "Start Y": [0.0] * 4,
            "End X": [1.0] * 4,
            "End Y": [1.0] * 4,
        }
    )
    data = MatchData(
        match_id="99", events=df, home_tracking=tracking, away_tracking=away
    )
    with pytest.warns(UserWarning, match="Periods \\[3\\].*2 event"):
        clips = EventExtractor().extract(data, "pass", infer_endpoints=False)
    assert len(clips) == 2
    assert all(c.metadata["period"] == 1 for c in clips)


def test_extract_event_idx_sequential_after_period_skip() -> None:
    """event_idx in returned clips must be 0-based and sequential
    even when earlier events were dropped due to missing tracking."""
    tracking = _make_tracking_with_periods([1])
    away = tracking.rename(columns={"Home_1_x": "Away_1_x"})
    df = pd.DataFrame(
        {
            "Team": ["Home", "Home", "Home"],
            "Type": ["pass", "pass", "pass"],
            "Subtype": ["success"] * 3,
            "Period": [3, 1, 1],  # period 3 first → skipped
            "Start Frame": [0, 1, 2],
            "Start Time [s]": [0.0, 1.0, 2.0],
            "End Frame": [1, 2, 3],
            "End Time [s]": [1.0, 2.0, 3.0],
            "From": ["Alice"] * 3,
            "To": ["Bob"] * 3,
            "Start X": [0.0] * 3,
            "Start Y": [0.0] * 3,
            "End X": [1.0] * 3,
            "End Y": [1.0] * 3,
        }
    )
    data = MatchData(
        match_id="99", events=df, home_tracking=tracking, away_tracking=away
    )
    with pytest.warns(UserWarning):
        clips = EventExtractor().extract(data, "pass", infer_endpoints=False)
    assert [c.event_idx for c in clips] == [0, 1]


def test_extract_no_warning_when_all_periods_covered() -> None:
    """No warning when tracking covers every period present in events."""
    tracking = _make_tracking_with_periods([1, 2, 3, 4])
    away = tracking.rename(columns={"Home_1_x": "Away_1_x"})
    df = pd.DataFrame(
        {
            "Team": ["Home"] * 4,
            "Type": ["pass"] * 4,
            "Subtype": ["success"] * 4,
            "Period": [1, 2, 3, 4],
            "Start Frame": [0, 5, 10, 15],
            "Start Time [s]": [0.0, 5.0, 10.0, 15.0],
            "End Frame": [1, 6, 11, 16],
            "End Time [s]": [1.0, 6.0, 11.0, 16.0],
            "From": ["Alice"] * 4,
            "To": ["Bob"] * 4,
            "Start X": [0.0] * 4,
            "Start Y": [0.0] * 4,
            "End X": [1.0] * 4,
            "End Y": [1.0] * 4,
        }
    )
    data = MatchData(
        match_id="99", events=df, home_tracking=tracking, away_tracking=away
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clips = EventExtractor().extract(data, "pass", infer_endpoints=False)
    assert len(clips) == 4


def test_extract_cross_infers_endpoints(
    home_tracking: pd.DataFrame,
    away_tracking: pd.DataFrame,
) -> None:
    df = pd.DataFrame(
        {
            "Team": ["Home", "Away"],
            "Type": ["cross", "pass"],
            "Subtype": ["success", "success"],
            "Period": [1, 1],
            "Start Frame": [5, 10],
            "Start Time [s]": [5.0, 10.0],
            "End Frame": [5, 15],
            "End Time [s]": [5.0, 15.0],
            "From": ["Alice", "Charlie"],
            "To": ["Bob", "Dave"],
            "Start X": [20.0, 30.0],
            "Start Y": [10.0, 5.0],
            "End X": [20.0, 30.0],
            "End Y": [10.0, 5.0],
        }
    )
    data = MatchData(
        match_id="99",
        events=df,
        home_tracking=home_tracking,
        away_tracking=away_tracking,
    )
    clips = EventExtractor().extract(data, "cross")
    # original PFF end unchanged; inferred from next event (pass at t=10.0)
    assert clips[0].metadata["end_x"] == pytest.approx(20.0)
    assert clips[0].metadata["inferred_end_x"] == pytest.approx(30.0)
    assert clips[0].metadata["inferred_end_y"] == pytest.approx(5.0)

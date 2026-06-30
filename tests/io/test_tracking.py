"""Tests for turf.io.tracking — build_possession_sequences_from_tracking."""

from __future__ import annotations

import pandas as pd
import pytest

from turf.io.tracking import build_possession_sequences_from_tracking


def _frames(
    rows: list[tuple[int, float, str, str | None]],
) -> pd.DataFrame:
    """Build minimal frames: (period, timestamp_sec, ball_state, ball_owning_team)."""
    return pd.DataFrame(
        rows,
        columns=["period", "timestamp_sec", "ball_state", "ball_owning_team"],
    )


class TestBuildPossessionSequencesFromTracking:
    def test_empty_returns_empty(self) -> None:
        seqs = build_possession_sequences_from_tracking(_frames([]))
        assert len(seqs) == 0

    def test_only_dead_frames_returns_empty(self) -> None:
        df = _frames([(1, 0.0, "dead", "Home"), (1, 0.033, "dead", "Away")])
        assert len(build_possession_sequences_from_tracking(df)) == 0

    def test_null_owning_team_excluded(self) -> None:
        df = _frames([(1, 0.0, "alive", None), (1, 0.033, "alive", "Home")])
        seqs = build_possession_sequences_from_tracking(df)
        assert len(seqs) == 1
        assert seqs.iloc[0]["team"] == "Home"

    def test_single_alive_frame_one_sequence(self) -> None:
        df = _frames([(1, 5.0, "alive", "Home")])
        assert len(build_possession_sequences_from_tracking(df)) == 1

    def test_consecutive_same_team_one_sequence(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "alive", "Home"),
            (1, 0.066, "alive", "Home"),
        ])
        assert len(build_possession_sequences_from_tracking(df)) == 1

    def test_team_change_two_sequences(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "alive", "Away"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert len(seqs) == 2
        assert list(seqs["team"]) == ["Home", "Away"]

    def test_dead_frames_between_same_team_not_split(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "dead",  "Home"),
            (1, 0.066, "alive", "Home"),
        ])
        assert len(build_possession_sequences_from_tracking(df)) == 1

    def test_period_change_creates_new_sequence(self) -> None:
        df = _frames([
            (1, 40.0, "alive", "Home"),
            (2,  0.0, "alive", "Home"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert len(seqs) == 2
        assert seqs.iloc[0]["period"] == 1
        assert seqs.iloc[1]["period"] == 2

    def test_start_end_time_from_alive_timestamps(self) -> None:
        df = _frames([
            (1, 0.033, "alive", "Home"),
            (1, 0.066, "alive", "Home"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert seqs.iloc[0]["start_time"] == pytest.approx(0.033)
        assert seqs.iloc[0]["end_time"] == pytest.approx(0.066)

    def test_duration_is_alive_timestamp_range(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "dead",  "Home"),
            (1, 0.066, "alive", "Home"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        # alive timestamps: [0.0, 0.066] → duration = 0.066
        assert seqs.iloc[0]["duration_sec"] == pytest.approx(0.066, abs=1e-3)

    def test_dead_ball_gap_not_attributed_to_either_team(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "alive", "Home"),
            (1, 0.066, "dead",  None),
            (1, 5.0,   "dead",  None),
            (1, 5.033, "alive", "Away"),
            (1, 5.066, "alive", "Away"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert len(seqs) == 2
        home = seqs[seqs["team"] == "Home"].iloc[0]
        away = seqs[seqs["team"] == "Away"].iloc[0]
        assert home["duration_sec"] == pytest.approx(0.033, abs=1e-3)
        assert away["duration_sec"] == pytest.approx(0.033, abs=1e-3)

    def test_n_events_is_alive_frame_count(self) -> None:
        df = _frames([
            (1, 0.0,   "alive", "Home"),
            (1, 0.033, "dead",  "Home"),
            (1, 0.066, "alive", "Home"),
            (1, 0.1,   "alive", "Home"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert seqs.iloc[0]["n_events"] == 3

    def test_output_columns(self) -> None:
        df = _frames([(1, 0.0, "alive", "Home")])
        seqs = build_possession_sequences_from_tracking(df)
        expected = {
            "period", "team", "start_time", "end_time", "duration_sec", "n_events"
        }
        assert set(seqs.columns) == expected

    def test_multiple_spells_same_team_different_periods(self) -> None:
        df = _frames([
            (1, 0.0,  "alive", "Away"),
            (1, 10.0, "alive", "Home"),
            (2, 0.0,  "alive", "Away"),
        ])
        seqs = build_possession_sequences_from_tracking(df)
        assert len(seqs) == 3
        assert list(seqs["team"]) == ["Away", "Home", "Away"]

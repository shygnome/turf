"""Tests for turf.possession — summarize_possession."""

from __future__ import annotations

import pandas as pd
import pytest

from turf.possession import summarize_possession


def _seqs(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """Build a minimal sequences DataFrame: (period, team, duration_sec)."""
    return pd.DataFrame(
        [{"period": p, "team": t, "duration_sec": d, "n_events": 1} for p, t, d in rows]
    )


class TestSummarizePossession:
    def test_output_columns(self) -> None:
        summary = summarize_possession(_seqs([(1, "Home", 10.0)]))
        assert {"period", "home_sec", "away_sec"}.issubset(set(summary.columns))

    def test_home_sec_sum(self) -> None:
        seqs = _seqs([(1, "Home", 100.0), (1, "Home", 50.0), (1, "Away", 80.0)])
        summary = summarize_possession(seqs)
        row = summary[summary["period"] == 1].iloc[0]
        assert row["home_sec"] == pytest.approx(150.0)

    def test_away_sec_sum(self) -> None:
        seqs = _seqs([(1, "Home", 100.0), (1, "Away", 80.0), (1, "Away", 20.0)])
        summary = summarize_possession(seqs)
        row = summary[summary["period"] == 1].iloc[0]
        assert row["away_sec"] == pytest.approx(100.0)

    def test_both_values_present_per_period(self) -> None:
        seqs = _seqs([(1, "Home", 50.0), (1, "Away", 40.0)])
        summary = summarize_possession(seqs)
        row = summary[summary["period"] == 1].iloc[0]
        assert "home_sec" in summary.columns
        assert "away_sec" in summary.columns
        assert row["home_sec"] > 0
        assert row["away_sec"] > 0

    def test_two_periods_two_rows(self) -> None:
        seqs = _seqs([
            (1, "Home", 100.0), (1, "Away", 80.0),
            (2, "Home", 90.0),  (2, "Away", 110.0),
        ])
        summary = summarize_possession(seqs)
        assert len(summary) == 2

    def test_per_period_values_independent(self) -> None:
        seqs = _seqs([
            (1, "Home", 60.0), (1, "Away", 50.0),
            (2, "Home", 40.0), (2, "Away", 70.0),
        ])
        summary = summarize_possession(seqs)
        p1 = summary[summary["period"] == 1].iloc[0]
        p2 = summary[summary["period"] == 2].iloc[0]
        assert p1["home_sec"] == pytest.approx(60.0)
        assert p2["home_sec"] == pytest.approx(40.0)
        assert p1["away_sec"] == pytest.approx(50.0)
        assert p2["away_sec"] == pytest.approx(70.0)

    def test_zero_possession_for_missing_team(self) -> None:
        seqs = _seqs([(1, "Home", 90.0)])
        summary = summarize_possession(seqs)
        row = summary[summary["period"] == 1].iloc[0]
        assert row["away_sec"] == pytest.approx(0.0)

    def test_empty_sequences_returns_empty(self) -> None:
        seqs = pd.DataFrame(
            columns=["period", "team", "duration_sec", "n_events"]
        )
        summary = summarize_possession(seqs)
        assert len(summary) == 0

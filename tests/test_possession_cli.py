"""CLI integration tests for turf dataset possession."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from turf.cli import app

runner = CliRunner()

DATASET_ID = "pff/fifa-wc-2022"
MATCH_ID = "3812"


def _make_frames() -> pd.DataFrame:
    """Minimal tracking frames yielding 3 sequences: Home, Away, Home."""
    return pd.DataFrame({
        "period":           [1,      1,      1,      1,      1,      1],
        "timestamp_sec":    [0.0,    1.96,   2.0,    3.96,   4.0,    4.96],
        "ball_state":       ["alive", "alive", "alive", "alive", "alive", "alive"],
        "ball_owning_team": ["Home", "Home", "Away", "Away", "Home", "Home"],
    })


@pytest.fixture()
def poss_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Returns (data_root, output_root); tracking data is mocked."""
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    return data_root, output_root


def _invoke(data: Path, out: Path, match_id: str = MATCH_ID) -> object:
    with patch(
        "turf.possession.load_tracking_frames", return_value=_make_frames()
    ):
        return runner.invoke(
            app,
            [
                "dataset", "possession", DATASET_ID, match_id,
                "--data-root", str(data),
                "--output-root", str(out),
            ],
        )


# ── exit codes ────────────────────────────────────────────────────────────────


class TestPossessionExitCode:
    def test_success(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        assert _invoke(data, out).exit_code == 0

    def test_unknown_dataset(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        result = runner.invoke(
            app,
            ["dataset", "possession", "bad/id", MATCH_ID,
             "--data-root", str(data), "--output-root", str(out)],
        )
        assert result.exit_code != 0

    def test_kloppy_error_causes_nonzero_exit(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        with patch(
            "turf.possession.load_tracking_frames",
            side_effect=FileNotFoundError("raw file not found"),
        ):
            result = runner.invoke(
                app,
                ["dataset", "possession", DATASET_ID, MATCH_ID,
                 "--data-root", str(data), "--output-root", str(out)],
            )
        assert result.exit_code != 0


# ── output files ──────────────────────────────────────────────────────────────


class TestPossessionOutputFiles:
    def test_creates_sequences_csv(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        assert (out / DATASET_ID / MATCH_ID / "possession_sequences.csv").exists()

    def test_creates_summary_csv(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        assert (out / DATASET_ID / MATCH_ID / "possession_summary.csv").exists()


# ── sequences csv content ─────────────────────────────────────────────────────


class TestPossessionSequencesContent:
    def test_sequences_has_required_columns(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_sequences.csv")
        required = {
            "period", "team", "start_time", "end_time", "duration_sec", "n_events"
        }
        assert required.issubset(set(df.columns))

    def test_sequences_team_values(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_sequences.csv")
        assert set(df["team"].unique()).issubset({"Home", "Away"})

    def test_sequences_count(self, poss_dirs: tuple) -> None:
        # frames: Home Home Away Away Home Home → 3 sequences
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_sequences.csv")
        assert len(df) == 3

    def test_sequences_order(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_sequences.csv")
        assert list(df["team"]) == ["Home", "Away", "Home"]


# ── summary csv content ───────────────────────────────────────────────────────


class TestPossessionSummaryContent:
    def test_summary_has_required_columns(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_summary.csv")
        assert {"period", "home_sec", "away_sec"}.issubset(set(df.columns))

    def test_summary_has_both_home_and_away(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_summary.csv")
        row = df[df["period"] == 1].iloc[0]
        assert row["home_sec"] > 0
        assert row["away_sec"] > 0

    def test_summary_home_sec_value(self, poss_dirs: tuple) -> None:
        # Home frames: [0.0→1.96] + [4.0→4.96]  durations: 1.96 + 0.96 = 2.92
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_summary.csv")
        row = df[df["period"] == 1].iloc[0]
        assert row["home_sec"] == pytest.approx(1.96 + (4.96 - 4.0), rel=1e-3)

    def test_summary_away_sec_value(self, poss_dirs: tuple) -> None:
        # Away frames: [2.0→3.96]  duration: 1.96
        data, out = poss_dirs
        _invoke(data, out)
        df = pd.read_csv(out / DATASET_ID / MATCH_ID / "possession_summary.csv")
        row = df[df["period"] == 1].iloc[0]
        assert row["away_sec"] == pytest.approx(3.96 - 2.0, rel=1e-3)

    def test_summary_cli_output_mentions_match(self, poss_dirs: tuple) -> None:
        data, out = poss_dirs
        result = _invoke(data, out)
        assert MATCH_ID in result.output

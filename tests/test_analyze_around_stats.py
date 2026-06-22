"""Tests for turf analyze leak around-stats CLI command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from turf.cli import app

runner = CliRunner()

DATASET_ID = "pff/fifa-wc-2022"

# ── fixture helpers ───────────────────────────────────────────────────────────

def _make_labeled(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal labeled_metadata.csv DataFrame."""
    defaults = {
        "event_idx": 0,
        "start_time": 0.0,
        "end_time": 1.0,
        "start_x": 0.0,
        "start_y": 0.0,
        "end_x": 1.0,
        "end_y": 1.0,
        "from_player": "A",
        "to_player": "B",
        "subtype": "success",
        "period": 1,
        "lines_broken_count": 0,
        "lines_broken": "[]",
        "location_after_break": "",
    }
    records = []
    for i, r in enumerate(rows):
        row = {**defaults, "event_idx": i, **r}
        records.append(row)
    return pd.DataFrame(records)


def _write_labeled(out_dir: Path, match_id: str, df: pd.DataFrame) -> None:
    pass_dir = out_dir / DATASET_ID / match_id / "pass"
    pass_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(pass_dir / "labeled_metadata.csv", index=False)


def _write_goals(path: Path, rows: list[tuple[int, str]]) -> None:
    """Write a minimal goals CSV: [(match_id, scoring_team), ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("match_id,scoring_team,period,minute,added_time\n")
        for match_id, scoring_team in rows:
            f.write(f"{match_id},{scoring_team},1,30,0\n")


def _write_possession_summary(
    out_dir: Path, match_id: str, home_sec: float, away_sec: float
) -> None:
    """Write a minimal possession_summary.csv (single-period, equal possession)."""
    poss_dir = out_dir / DATASET_ID / match_id
    poss_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"period": 1, "home_sec": home_sec, "away_sec": away_sec}]
    ).to_csv(poss_dir / "possession_summary.csv", index=False)


# ── fixture: two matches with known outcomes ──────────────────────────────────
#
# Match 3001 — Home wins 1-0.  10 passes each team (equal possession = 45 min).
#   Home: 4 × [Around]              → Sc1: 4, Sc2: 4
#         2 × [Through, Around]     → Sc1: +2, Sc2: +0
#         4 × not breaking
#   Away: 1 × [Around]              → Sc1: 1, Sc2: 1
#         9 × not breaking
#   Sc1: Home 6 > Away 1 → Home more → Home wins ✓
#   Sc2: Home 4 > Away 1 → Home more → Home wins ✓
#
# Match 3002 — Away wins 1-0.  10 passes each team.
#   Home: 1 × [Around]              → Sc1: 1, Sc2: 1
#         9 × not breaking
#   Away: 3 × [Around]              → Sc1: 3, Sc2: 3
#         1 × [Through, Around]     → Sc1: +1, Sc2: +0
#         6 × not breaking
#   Sc1: Away 4 > Home 1 → Away more → Away wins ✓
#   Sc2: Away 3 > Home 1 → Away more → Away wins ✓
#
# Both scenarios: win rate = 2/2 = 100 %.
#
# Possession (equal passes) → poss_min = 45 for each team.
# around_per_30 = breaks / (45/30) = breaks / 1.5
#
# Scenario 1 avg diff:
#   match 3001: (6-1)/1.5 = 3.33   match 3002: (4-1)/1.5 = 2.0   avg = 2.67
# Scenario 2 avg diff:
#   match 3001: (4-1)/1.5 = 2.0    match 3002: (3-1)/1.5 = 1.33  avg = 1.67


_ARN = "['Around']"
_THA = "['Through', 'Around']"
_NON = "[]"


def _rows(team: str, breaking: bool, dirs: str, n: int) -> list[dict]:
    row = {"team": team, "is_line_breaking": breaking, "direction_per_line": dirs}
    return [row] * n


def _build_match_3001() -> pd.DataFrame:
    rows = (
        _rows("Home", True,  _ARN, 4)
        + _rows("Home", True,  _THA, 2)
        + _rows("Home", False, _NON, 4)
        + _rows("Away", True,  _ARN, 1)
        + _rows("Away", False, _NON, 9)
    )
    return _make_labeled(rows)


def _build_match_3002() -> pd.DataFrame:
    rows = (
        _rows("Home", True,  _ARN, 1)
        + _rows("Home", False, _NON, 9)
        + _rows("Away", True,  _ARN, 3)
        + _rows("Away", True,  _THA, 1)
        + _rows("Away", False, _NON, 6)
    )
    return _make_labeled(rows)


@pytest.fixture()
def around_stats_dir(tmp_path: Path) -> tuple[Path, Path]:
    """Returns (output_root, goals_csv).

    Possession: 2700s each team (45 min) — keeps the same per-30 math as before.
    """
    out = tmp_path / "output"
    _write_labeled(out, "3001", _build_match_3001())
    _write_labeled(out, "3002", _build_match_3002())
    _write_possession_summary(out, "3001", home_sec=2700.0, away_sec=2700.0)
    _write_possession_summary(out, "3002", home_sec=2700.0, away_sec=2700.0)

    goals = tmp_path / "goals.csv"
    _write_goals(goals, [(3001, "Home"), (3002, "Away")])

    return out, goals


def _invoke(out: Path, goals: Path, extra: list[str] | None = None) -> object:
    args = [
        "analyze", "leak", "around-stats", DATASET_ID,
        "--output-root", str(out),
        "--goals", str(goals),
    ] + (extra or [])
    return runner.invoke(app, args)


# ── exit code ─────────────────────────────────────────────────────────────────


class TestAroundStatsExitCode:
    def test_success(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert result.exit_code == 0

    def test_unknown_dataset(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        bad = runner.invoke(
            app,
            ["analyze", "leak", "around-stats", "bad/id",
             "--output-root", str(out), "--goals", str(goals)],
        )
        assert bad.exit_code != 0

    def test_missing_goals_csv(self, around_stats_dir: tuple) -> None:
        out, _ = around_stats_dir
        result = runner.invoke(
            app,
            ["analyze", "leak", "around-stats", DATASET_ID,
             "--output-root", str(out), "--goals", str(out / "nonexistent.csv")],
        )
        assert result.exit_code != 0

    def test_no_labeled_files(self, tmp_path: Path) -> None:
        empty_out = tmp_path / "empty"
        empty_out.mkdir()
        goals = tmp_path / "goals.csv"
        _write_goals(goals, [])
        result = runner.invoke(
            app,
            ["analyze", "leak", "around-stats", DATASET_ID,
             "--output-root", str(empty_out), "--goals", str(goals)],
        )
        assert result.exit_code != 0


# ── output structure ──────────────────────────────────────────────────────────


class TestAroundStatsOutput:
    def test_shows_scenario_0(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "Scenario 0" in result.output

    def test_shows_scenario_1(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "Scenario 1" in result.output

    def test_shows_scenario_2(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "Scenario 2" in result.output

    def test_shows_dataset_id(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert DATASET_ID in result.output

    def test_shows_match_count(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "2" in result.output


# ── win-rate calculation ──────────────────────────────────────────────────────


class TestAroundStatsValues:
    def test_win_rate_100_pct_all_scenarios(self, around_stats_dir: tuple) -> None:
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        # All three scenarios: the team with more breaks always wins
        assert result.output.count("100.0%") == 3

    def test_scenario1_avg_diff_approx_2_67(self, around_stats_dir: tuple) -> None:
        """Sc1 diff = (5/1.5 + 3/1.5) / 2 = 2.67"""
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "2.67" in result.output

    def test_scenario2_avg_diff_approx_1_67(self, around_stats_dir: tuple) -> None:
        """Sc2 diff = (3/1.5 + 2/1.5) / 2 = 1.67"""
        out, goals = around_stats_dir
        result = _invoke(out, goals)
        assert "1.67" in result.output

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from turf.cli import app
from turf.match_loader import MatchData, MatchLoader

runner = CliRunner()

METADATA_COLS = [
    "match_id",
    "home_team_id",
    "home_team_name",
    "home_team_short_name",
    "away_team_id",
    "away_team_name",
    "away_team_short_name",
    "date",
    "stadium",
]


def _write_metadata_csv(base: Path) -> None:
    # Rows are intentionally unsorted (9500, 10502, 1000) so tests can assert
    # that the output is sorted numerically: 1000 < 9500 < 10502.
    # Note: string sort would give 1000 < 10502 < 9500 — a different order.
    pd.DataFrame(
        [
            {
                "match_id": 9500,
                "home_team_id": 3,
                "home_team_name": "Germany",
                "home_team_short_name": "GER",
                "away_team_id": 4,
                "away_team_name": "Spain",
                "away_team_short_name": "ESP",
                "date": "2022-12-06T15:00:00",
                "stadium": "Stadium B",
            },
            {
                "match_id": 10502,
                "home_team_id": 1,
                "home_team_name": "Netherlands",
                "home_team_short_name": "NED",
                "away_team_id": 2,
                "away_team_name": "United States",
                "away_team_short_name": "USA",
                "date": "2022-12-03T15:00:00",
                "stadium": "Khalifa International Stadium",
            },
            {
                "match_id": 1000,
                "home_team_id": 5,
                "home_team_name": "France",
                "home_team_short_name": "FRA",
                "away_team_id": 6,
                "away_team_name": "England",
                "away_team_short_name": "ENG",
                "date": "2022-11-28T15:00:00",
                "stadium": "Stadium A",
            },
        ]
    ).to_csv(base / "metadata.csv", index=False)


# ---------------------------------------------------------------------------
# Fixtures — synthetic preprocessed layout
# ---------------------------------------------------------------------------


@pytest.fixture()
def preprocessed_root(tmp_path: Path) -> Path:
    """
    Build a minimal preprocessed directory tree for one dataset + two matches.

    Layout mirrors what openstarlab-preprocessing produces:
        <root>/preprocessed/pff/fifa-wc-2022/
            event/          event_data_10502.csv
            home_tracking/  home_tracking_10502.csv
            away_tracking/  away_tracking_10502.csv
    """
    dataset_id = "pff/fifa-wc-2022"
    base = tmp_path / "preprocessed" / Path(dataset_id)

    event_cols = [
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ]
    tracking_home_cols = [
        "Period",
        "Time [s]",
        "Home_1_x",
        "Home_1_y",
        "ball_x",
        "ball_y",
    ]  # noqa: E501
    tracking_away_cols = [
        "Period",
        "Time [s]",
        "Away_1_x",
        "Away_1_y",
        "ball_x",
        "ball_y",
    ]  # noqa: E501

    def _write(subdir: str, filename: str, cols: list[str], rows: int = 3) -> None:
        d = base / subdir
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({c: range(rows) for c in cols}).to_csv(d / filename, index=False)

    for mid in [9500, 10502, 1000]:
        _write("event", f"event_data_{mid}.csv", event_cols, rows=5)
        _write("home_tracking", f"home_tracking_{mid}.csv", tracking_home_cols, rows=10)
        _write("away_tracking", f"away_tracking_{mid}.csv", tracking_away_cols, rows=10)

    return tmp_path


# ---------------------------------------------------------------------------
# MatchLoader construction
# ---------------------------------------------------------------------------


def test_loader_resolves_preprocessed_path(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    expected = preprocessed_root / "preprocessed" / "pff" / "fifa-wc-2022"
    assert loader.preprocessed_path == expected


# ---------------------------------------------------------------------------
# MatchLoader.load — happy path
# ---------------------------------------------------------------------------


def test_load_returns_match_data(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert isinstance(result, MatchData)


def test_load_match_id_stored(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert result.match_id == "10502"


def test_load_events_is_dataframe(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert isinstance(result.events, pd.DataFrame)


def test_load_events_row_count(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert len(result.events) == 5


def test_load_home_tracking_is_dataframe(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert isinstance(result.home_tracking, pd.DataFrame)


def test_load_away_tracking_is_dataframe(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert isinstance(result.away_tracking, pd.DataFrame)


def test_load_tracking_row_count(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    assert len(result.home_tracking) == 10
    assert len(result.away_tracking) == 10


def test_load_events_expected_columns(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    result = loader.load("10502")
    expected = {
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    }
    assert expected.issubset(set(result.events.columns))


# ---------------------------------------------------------------------------
# MatchLoader.load — error cases
# ---------------------------------------------------------------------------


def test_load_missing_event_file_raises(preprocessed_root: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=preprocessed_root)
    with pytest.raises(FileNotFoundError):
        loader.load("99999")


def test_load_preprocessed_dir_missing_raises(tmp_path: Path) -> None:
    loader = MatchLoader(dataset_id="pff/fifa-wc-2022", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load("10502")


# ---------------------------------------------------------------------------
# CLI — turf match load
# ---------------------------------------------------------------------------


def test_cli_match_load_exit_code(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "load", "pff/fifa-wc-2022", "10502"])
    assert result.exit_code == 0


def test_cli_match_load_shows_match_id(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "load", "pff/fifa-wc-2022", "10502"])
    assert "10502" in result.output


def test_cli_match_load_shows_row_counts(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "load", "pff/fifa-wc-2022", "10502"])
    assert "5" in result.output  # events rows
    assert "10" in result.output  # tracking rows


def test_cli_match_load_unknown_dataset_exits_nonzero(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "load", "unknown/dataset", "10502"])
    assert result.exit_code != 0


def test_cli_match_load_unknown_match_exits_nonzero(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "load", "pff/fifa-wc-2022", "99999"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — turf match ls (with metadata.csv)
# ---------------------------------------------------------------------------


@pytest.fixture()
def preprocessed_root_with_metadata(preprocessed_root: Path) -> Path:
    base = preprocessed_root / "preprocessed" / "pff" / "fifa-wc-2022"
    _write_metadata_csv(base)
    return preprocessed_root


def test_cli_match_ls_with_metadata_exit_code(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert result.exit_code == 0


def test_cli_match_ls_shows_match_id(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "10502" in result.output


def test_cli_match_ls_shows_team_names(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "Netherlands" in result.output
    assert "United States" in result.output


def test_cli_match_ls_shows_date(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "2022-12-03" in result.output


# ---------------------------------------------------------------------------
# CLI — turf match ls (fallback: no metadata.csv)
# ---------------------------------------------------------------------------


def test_cli_match_ls_fallback_exit_code(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert result.exit_code == 0


def test_cli_match_ls_fallback_shows_match_id(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "10502" in result.output


def test_cli_match_ls_fallback_has_table_header(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "MATCH_ID" in result.output
    assert "HOME vs AWAY" in result.output
    assert "DATE" in result.output


def test_cli_match_ls_fallback_shows_placeholder(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert "n/a" in result.output


# ---------------------------------------------------------------------------
# CLI — turf match ls error cases
# ---------------------------------------------------------------------------


def test_cli_match_ls_unknown_dataset_exits_nonzero(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "unknown/dataset"])
    assert result.exit_code != 0


def test_cli_match_ls_not_prepared_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — turf match ls sort order (numeric ascending)
# ---------------------------------------------------------------------------


def test_cli_match_ls_sorted_numerically_with_metadata(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    lines = [ln for ln in result.output.splitlines() if ln and ln[0].isdigit()]
    ids = [int(ln.split()[0]) for ln in lines]
    assert ids == sorted(ids)


def test_cli_match_ls_sorted_numerically_fallback(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["match", "ls", "pff/fifa-wc-2022"])
    lines = [ln for ln in result.output.splitlines() if ln and ln[0].isdigit()]
    ids = [int(ln.split()[0]) for ln in lines]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# CLI — turf match ls pagination
# ---------------------------------------------------------------------------


def test_cli_match_ls_per_page_limits_rows(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(
        app, ["match", "ls", "pff/fifa-wc-2022", "--per-page", "2"]
    )
    lines = [ln for ln in result.output.splitlines() if ln and ln[0].isdigit()]
    assert len(lines) == 2


def test_cli_match_ls_page_2_shows_different_rows(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    p1 = runner.invoke(
        app, ["match", "ls", "pff/fifa-wc-2022", "--per-page", "2", "--page", "1"]
    )
    p2 = runner.invoke(
        app, ["match", "ls", "pff/fifa-wc-2022", "--per-page", "2", "--page", "2"]
    )
    lines1 = [ln for ln in p1.output.splitlines() if ln and ln[0].isdigit()]
    lines2 = [ln for ln in p2.output.splitlines() if ln and ln[0].isdigit()]
    assert lines1 != lines2


def test_cli_match_ls_shows_page_footer(
    preprocessed_root_with_metadata: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root_with_metadata)
    result = runner.invoke(
        app, ["match", "ls", "pff/fifa-wc-2022", "--per-page", "2"]
    )
    assert "Page 1" in result.output


def test_cli_match_ls_fallback_pagination(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.match.get_root", lambda: preprocessed_root)
    result = runner.invoke(
        app, ["match", "ls", "pff/fifa-wc-2022", "--per-page", "2"]
    )
    lines = [ln for ln in result.output.splitlines() if ln and ln[0].isdigit()]
    assert len(lines) == 2

"""CLI integration tests for turf ranking fetch."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from turf.cli import app
from turf.ranking import RankingEntry

runner = CliRunner()

_ENTRIES = [
    RankingEntry(
        rank=1,
        name="Brazil",
        country_code="BRA",
        confederation="CONMEBOL",
        total_points=1841.30,
        previous_rank=1,
        previous_points=1837.56,
        points_change=3.74,
        last_update_date="2022-10-06T00:00:00Z",
    ),
    RankingEntry(
        rank=2,
        name="Belgium",
        country_code="BEL",
        confederation="UEFA",
        total_points=1816.71,
        previous_rank=2,
        previous_points=1821.92,
        points_change=-5.21,
        last_update_date="2022-10-06T00:00:00Z",
    ),
]


class TestRankingFetchCLI:
    def test_exit_code(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792"])
        assert result.exit_code == 0

    def test_shows_team_names(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792"])
        assert "Brazil" in result.output
        assert "Belgium" in result.output

    def test_shows_ranks(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792"])
        assert "1" in result.output
        assert "2" in result.output

    def test_shows_points(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792"])
        assert "1841" in result.output

    def test_top_limits_output(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792", "--top", "1"])
        assert "Brazil" in result.output
        assert "Belgium" not in result.output

    def test_csv_output(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792", "--csv"])
        assert "rank,name,country_code" in result.output

    def test_csv_first_row(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=_ENTRIES):
            result = runner.invoke(app, ["ranking", "fetch", "id13792", "--csv"])
        assert "1,Brazil,BRA" in result.output

    def test_empty_exits_nonzero(self) -> None:
        with patch("turf.ranking.fetch_rankings", return_value=[]):
            result = runner.invoke(app, ["ranking", "fetch", "id99999"])
        assert result.exit_code != 0

    def test_api_error_exits_nonzero(self) -> None:
        with patch("turf.ranking.fetch_rankings", side_effect=ValueError("HTTP 404")):
            result = runner.invoke(app, ["ranking", "fetch", "id13792"])
        assert result.exit_code != 0

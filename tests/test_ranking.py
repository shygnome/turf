"""Tests for turf.ranking — fetch_rankings and get_wc22_rankings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turf.ranking import RankingEntry, fetch_rankings, get_wc22_rankings

# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_urlopen(rankings: list[dict]) -> MagicMock:
    body = json.dumps({"rankings": rankings}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


_ENTRY_1 = {
    "rankingItem": {
        "rank": 1,
        "name": "Brazil",
        "countryCode": "BRA",
        "totalPoints": 1841.30,
        "previousRank": 1,
    },
    "previousPoints": 1837.56,
    "lastUpdateDate": "2022-10-06T00:00:00Z",
    "tag": {"text": "CONMEBOL"},
}

_ENTRY_2 = {
    "rankingItem": {
        "rank": 2,
        "name": "Belgium",
        "countryCode": "BEL",
        "totalPoints": 1816.71,
        "previousRank": 2,
    },
    "previousPoints": 1821.92,
    "lastUpdateDate": "2022-10-06T00:00:00Z",
    "tag": {"text": "UEFA"},
}


class TestFetchRankings:
    def test_returns_list(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert isinstance(result, list)

    def test_entry_count(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert len(result) == 2

    def test_entry_type(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert all(isinstance(e, RankingEntry) for e in result)

    def test_rank(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_name(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert result[0].name == "Brazil"
        assert result[1].name == "Belgium"

    def test_total_points(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert result[0].total_points == pytest.approx(1841.30)

    def test_previous_points(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert result[0].previous_points == pytest.approx(1837.56)

    def test_points_change_positive(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert result[0].points_change == pytest.approx(3.74)

    def test_points_change_negative(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_2])):
            result = fetch_rankings("id13792")
        assert result[0].points_change == pytest.approx(-5.21)

    def test_country_code(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert result[0].country_code == "BRA"
        assert result[1].country_code == "BEL"

    def test_confederation(self) -> None:
        mock = _mock_urlopen([_ENTRY_1, _ENTRY_2])
        with patch("urllib.request.urlopen", return_value=mock):
            result = fetch_rankings("id13792")
        assert result[0].confederation == "CONMEBOL"
        assert result[1].confederation == "UEFA"

    def test_previous_rank(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert result[0].previous_rank == 1

    def test_last_update_date(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([_ENTRY_1])):
            result = fetch_rankings("id13792")
        assert result[0].last_update_date == "2022-10-06T00:00:00Z"

    def test_empty_date_id_returns_empty_list(self) -> None:
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([])):
            result = fetch_rankings("id99999")
        assert result == []

    def test_http_error_raises_value_error(self) -> None:
        from urllib.error import HTTPError

        err = HTTPError(None, 404, "Not Found", None, None)  # type: ignore[arg-type]
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(ValueError),
        ):
            fetch_rankings("id13792")

    def test_url_error_raises_connection_error(self) -> None:
        from urllib.error import URLError

        err = URLError("Connection refused")
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(ConnectionError),
        ):
            fetch_rankings("id13792")


class TestGetWc22Rankings:
    def _write_csv(self, path: Path, rows: list[tuple[int, str]]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "name", "country_code", "confederation",
                             "total_points", "previous_rank", "previous_points",
                             "points_change", "last_update_date"])
            for rank, code in rows:
                writer.writerow([rank, "Team", code, "UEFA",
                                 1000.0, rank, 990.0, 10.0, "2022-10-06T00:00:00Z"])

    def test_returns_dict(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ranking.csv"
        self._write_csv(csv_path, [(1, "BRA")])
        result = get_wc22_rankings(csv_path)
        assert isinstance(result, dict)

    def test_country_code_to_rank_mapping(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ranking.csv"
        self._write_csv(csv_path, [(1, "BRA"), (2, "BEL"), (3, "ARG")])
        result = get_wc22_rankings(csv_path)
        assert result["BRA"] == 1
        assert result["BEL"] == 2
        assert result["ARG"] == 3

    def test_rank_is_int(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ranking.csv"
        self._write_csv(csv_path, [(1, "BRA")])
        result = get_wc22_rankings(csv_path)
        assert isinstance(result["BRA"], int)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_wc22_rankings(tmp_path / "nonexistent.csv")

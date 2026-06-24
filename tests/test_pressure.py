"""Tests for turf.pressure — PFF JSON pressure lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turf.pressure import load_match_pressure


def _write_json(tmp_path: Path, events: list[dict]) -> Path:
    p = tmp_path / "events.json"
    p.write_text(json.dumps(events))
    return p


def _pass_event(
    period: int,
    clock: int,
    home_team: bool,
    pressure_type: str,
) -> dict:
    return {
        "gameEvents": {
            "period": period,
            "startGameClock": clock,
            "homeTeam": home_team,
        },
        "possessionEvents": {
            "possessionEventType": "PA",
            "pressureType": pressure_type,
        },
    }


def _non_pass_event(period: int, clock: int, home_team: bool) -> dict:
    return {
        "gameEvents": {
            "period": period,
            "startGameClock": clock,
            "homeTeam": home_team,
        },
        "possessionEvents": {
            "possessionEventType": "SH",
            "pressureType": "P",
        },
    }


# ── load_match_pressure ───────────────────────────────────────────────────────

def test_returns_only_pass_events(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [
        _pass_event(1, 10, True, "N"),
        _non_pass_event(1, 20, True),
    ])
    result = load_match_pressure(path)
    assert (1, 10, "Home") in result
    assert (1, 20, "Home") not in result


def test_pressure_type_n_is_false(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [_pass_event(1, 5, True, "N")])
    assert load_match_pressure(path)[(1, 5, "Home")] is False


@pytest.mark.parametrize("ptype", ["P", "L", "A"])
def test_pressure_types_pla_are_true(tmp_path: Path, ptype: str) -> None:
    path = _write_json(tmp_path, [_pass_event(1, 5, True, ptype)])
    assert load_match_pressure(path)[(1, 5, "Home")] is True


def test_home_team_maps_to_home(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [_pass_event(1, 0, True, "N")])
    result = load_match_pressure(path)
    assert (1, 0, "Home") in result
    assert (1, 0, "Away") not in result


def test_away_team_maps_to_away(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [_pass_event(1, 0, False, "P")])
    result = load_match_pressure(path)
    assert (1, 0, "Away") in result
    assert (1, 0, "Home") not in result


def test_period_is_preserved(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [
        _pass_event(1, 30, True, "N"),
        _pass_event(2, 30, True, "P"),
    ])
    result = load_match_pressure(path)
    assert result[(1, 30, "Home")] is False
    assert result[(2, 30, "Home")] is True


def test_multiple_events_both_teams(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [
        _pass_event(1, 0, True, "N"),
        _pass_event(1, 1, False, "P"),
        _pass_event(1, 2, True, "A"),
    ])
    result = load_match_pressure(path)
    assert result[(1, 0, "Home")] is False
    assert result[(1, 1, "Away")] is True
    assert result[(1, 2, "Home")] is True


def test_missing_possession_events_key(tmp_path: Path) -> None:
    ev = {
        "gameEvents": {"period": 1, "startGameClock": 5, "homeTeam": True},
        "possessionEvents": None,
    }
    path = _write_json(tmp_path, [ev])
    result = load_match_pressure(path)
    assert (1, 5, "Home") not in result


def test_missing_game_events_key(tmp_path: Path) -> None:
    ev = {
        "gameEvents": None,
        "possessionEvents": {"possessionEventType": "PA", "pressureType": "P"},
    }
    path = _write_json(tmp_path, [ev])
    # Should not raise; event is skipped (no period/clock to key on)
    result = load_match_pressure(path)
    assert len(result) == 0


def test_empty_json(tmp_path: Path) -> None:
    path = _write_json(tmp_path, [])
    assert load_match_pressure(path) == {}

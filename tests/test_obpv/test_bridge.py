"""Tests for obpv.bridge — wide-format tracking frames → GameState."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from obpv.bridge import frames_to_game_state


def _make_frames(
    n: int = 3,
    home_positions: dict[int, tuple[float, float]] | None = None,
    away_positions: dict[int, tuple[float, float]] | None = None,
    ball_xy: tuple[float, float] = (5.0, 3.0),
    dt: float = 0.04,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal wide-format home/away DataFrames with n frames."""
    home_positions = home_positions or {1: (10.0, 5.0), 2: (-5.0, -2.0)}
    away_positions = away_positions or {1: (-20.0, 0.0), 3: (30.0, 10.0)}

    def _build(prefix: str, positions: dict[int, tuple[float, float]]) -> pd.DataFrame:
        rows = []
        for i in range(n):
            row: dict[str, object] = {
                "frame": i,
                "Period": 1,
                "Time [s]": i * dt,
                "ball_x": ball_xy[0],
                "ball_y": ball_xy[1],
            }
            for num, (x, y) in positions.items():
                row[f"{prefix}_{num}_x"] = x + i * 0.1
                row[f"{prefix}_{num}_y"] = y
            rows.append(row)
        return pd.DataFrame(rows)

    return _build("Home", home_positions), _build("Away", away_positions)


class TestFramesToGameState:
    def test_ball_position_frame0(self) -> None:
        home, away = _make_frames(ball_xy=(7.5, -2.0))
        state = frames_to_game_state(home, away, 0, "Home", 1)
        assert state.ball_position == pytest.approx([7.5, -2.0])

    def test_player_count_skips_nan(self) -> None:
        home, away = _make_frames(
            home_positions={1: (10.0, 0.0), 2: (5.0, 0.0)},
            away_positions={1: (-10.0, 0.0)},
        )
        state = frames_to_game_state(home, away, 0, "Home", 1)
        assert len(state.players) == 3

    def test_attacking_team_separation(self) -> None:
        home, away = _make_frames(
            home_positions={1: (10.0, 0.0)},
            away_positions={1: (-10.0, 0.0)},
        )
        state = frames_to_game_state(home, away, 0, "Home", 1)
        assert len(state.attacking_players) == 1
        assert len(state.defending_players) == 1
        assert state.attacking_players[0].team_id == "Home"
        assert state.defending_players[0].team_id == "Away"

    def test_attacking_team_away(self) -> None:
        home, away = _make_frames(
            home_positions={1: (10.0, 0.0)},
            away_positions={1: (-10.0, 0.0)},
        )
        state = frames_to_game_state(home, away, 0, "Away", -1)
        assert state.attacking_players[0].team_id == "Away"
        assert state.defending_players[0].team_id == "Home"

    def test_velocity_nonzero_with_multiple_frames(self) -> None:
        # Players move +0.1 in x per frame; dt=0.04 → vx ≈ 2.5 m/s
        home, away = _make_frames(n=3, dt=0.04)
        state = frames_to_game_state(home, away, 0, "Home", 1)
        home_player = state.attacking_players[0]
        assert abs(home_player.velocity[0]) == pytest.approx(2.5, rel=1e-3)

    def test_velocity_zero_single_frame(self) -> None:
        home, away = _make_frames(n=1)
        state = frames_to_game_state(home, away, 0, "Home", 1)
        for p in state.players:
            assert p.velocity == pytest.approx([0.0, 0.0])

    def test_last_frame_uses_backward_difference(self) -> None:
        home, away = _make_frames(n=3, dt=0.04)
        state_last = frames_to_game_state(home, away, 2, "Home", 1)
        home_player = state_last.attacking_players[0]
        # backward diff: same Δx/Δt = 2.5 m/s
        assert abs(home_player.velocity[0]) == pytest.approx(2.5, rel=1e-3)

    def test_attacking_direction_in_metadata_positive(self) -> None:
        home, away = _make_frames()
        state = frames_to_game_state(home, away, 0, "Home", 1)
        assert state.metadata["attacking_direction"] == "left_to_right"

    def test_attacking_direction_in_metadata_negative(self) -> None:
        home, away = _make_frames()
        state = frames_to_game_state(home, away, 0, "Home", -1)
        assert state.metadata["attacking_direction"] == "right_to_left"

    def test_nan_player_columns_skipped(self) -> None:
        home, away = _make_frames(
            home_positions={1: (10.0, 0.0)},
            away_positions={1: (-10.0, 0.0)},
        )
        # Manually insert a NaN player column
        home["Home_5_x"] = float("nan")
        home["Home_5_y"] = float("nan")
        state = frames_to_game_state(home, away, 0, "Home", 1)
        # NaN player should be skipped; still 2 total
        assert len(state.players) == 2

    def test_player_position_correct(self) -> None:
        home, away = _make_frames(
            n=1,
            home_positions={1: (15.0, -3.0)},
            away_positions={},
        )
        state = frames_to_game_state(home, away, 0, "Home", 1)
        assert state.attacking_players[0].position == pytest.approx([15.0, -3.0])

"""Tests for obpv._state â€” GameState and PlayerState."""

from __future__ import annotations

import numpy as np
import pytest

from obpv._state import GameState, PlayerState


def _player(team: str = "Home", x: float = 10.0, y: float = 5.0) -> PlayerState:
    return PlayerState(
        player_id=f"{team}_1",
        team_id=team,
        position=np.array([x, y]),
        velocity=np.array([0.0, 0.0]),
    )


def _minimal_state(ball_x: float = 0.0, ball_y: float = 0.0) -> GameState:
    return GameState(
        ball_position=np.array([ball_x, ball_y]),
        players=[_player("Home"), _player("Away", -10.0, -5.0)],
        attacking_team_id="Home",
        defending_team_id="Away",
    )


class TestPlayerState:
    def test_position_stored_as_float_array(self) -> None:
        p = _player()
        assert p.position.dtype == float
        assert p.position.shape == (2,)

    def test_velocity_defaults_to_zero_when_none(self) -> None:
        p = PlayerState(
            player_id="x",
            team_id="Home",
            position=np.array([0.0, 0.0]),
            velocity=None,  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(p.velocity, [0.0, 0.0])

    def test_wrong_position_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            PlayerState(
                player_id="x",
                team_id="Home",
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.array([0.0, 0.0]),
            )

    def test_xyz_properties(self) -> None:
        p = _player(x=3.0, y=-7.0)
        assert p.x == pytest.approx(3.0)
        assert p.y == pytest.approx(-7.0)


class TestGameState:
    def test_ball_position_stored_as_float(self) -> None:
        s = _minimal_state(5.0, -3.0)
        assert s.ball_position.dtype == float
        assert s.ball_x == pytest.approx(5.0)
        assert s.ball_y == pytest.approx(-3.0)

    def test_empty_players_raises(self) -> None:
        with pytest.raises(ValueError):
            GameState(
                ball_position=np.array([0.0, 0.0]),
                players=[],
            )

    def test_wrong_ball_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            GameState(
                ball_position=np.array([0.0, 0.0, 0.0]),
                players=[_player()],
            )

    def test_attacking_players_filtered_by_team(self) -> None:
        s = _minimal_state()
        att = s.attacking_players
        assert all(p.team_id == "Home" for p in att)

    def test_defending_players_filtered_by_team(self) -> None:
        s = _minimal_state()
        def_ = s.defending_players
        assert all(p.team_id == "Away" for p in def_)

    def test_players_for_team(self) -> None:
        s = GameState(
            ball_position=np.array([0.0, 0.0]),
            players=[_player("Home"), _player("Home", 5.0, 0.0), _player("Away")],
            attacking_team_id="Home",
            defending_team_id="Away",
        )
        assert len(s.players_for_team("Home")) == 2
        assert len(s.players_for_team("Away")) == 1


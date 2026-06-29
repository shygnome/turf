"""Tests for obpv.pass_obpv â€” compute_pass_obpv."""

from __future__ import annotations

import numpy as np

from obpv._grid import PitchGrid
from obpv._state import GameState, PlayerState
from obpv.analyzer import OBPVAnalyzer
from obpv.pass_obpv import compute_pass_obpv


def _grid() -> PitchGrid:
    return PitchGrid.from_resolution(n_x=6, n_y=4)


def _state(ball_x: float, ball_y: float, direction: str = "left_to_right") -> GameState:
    return GameState(
        ball_position=np.array([ball_x, ball_y]),
        players=[
            PlayerState("att1", "Home", np.array([10.0, 3.0]), np.array([1.0, 0.0])),
            PlayerState("att2", "Home", np.array([20.0, -3.0]), np.array([0.0, 0.0])),
            PlayerState("def1", "Away", np.array([-5.0, 3.0]), np.array([0.0, 0.0])),
            PlayerState("def2", "Away", np.array([-15.0, -3.0]), np.array([0.0, 0.0])),
        ],
        attacking_team_id="Home",
        defending_team_id="Away",
        metadata={"attacking_direction": direction},
    )


class TestComputePassOBPV:
    def test_returns_float(self) -> None:
        grid = _grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        gain = compute_pass_obpv(
            start_state=_state(-10.0, 0.0),
            end_state=_state(20.0, 0.0),
            analyzer=analyzer,
        )
        assert isinstance(gain, float)

    def test_forward_pass_generally_positive(self) -> None:
        grid = _grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        # Pass from deep own half to final third â€” should increase threat
        gain = compute_pass_obpv(
            start_state=_state(-30.0, 0.0),
            end_state=_state(30.0, 0.0),
            analyzer=analyzer,
        )
        assert gain > 0.0

    def test_same_state_gives_zero(self) -> None:
        grid = _grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        state = _state(0.0, 0.0)
        gain = compute_pass_obpv(
            start_state=state,
            end_state=state,
            analyzer=analyzer,
        )
        assert gain == 0.0

    def test_backward_pass_generally_negative(self) -> None:
        grid = _grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        gain = compute_pass_obpv(
            start_state=_state(30.0, 0.0),
            end_state=_state(-30.0, 0.0),
            analyzer=analyzer,
        )
        assert gain < 0.0


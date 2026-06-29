"""Tests for obpv._pitch_control â€” PPCFPitchControlModel."""

from __future__ import annotations

import numpy as np
import pytest

from obpv._grid import PitchGrid
from obpv._pitch_control import PPCFParameters, PPCFPitchControlModel
from obpv._state import GameState, PlayerState


def _small_grid() -> PitchGrid:
    return PitchGrid.from_resolution(n_x=6, n_y=4)


def _two_player_state(att_x: float = 10.0, def_x: float = -10.0) -> GameState:
    return GameState(
        ball_position=np.array([0.0, 0.0]),
        players=[
            PlayerState("att", "Home", np.array([att_x, 0.0]), np.array([0.0, 0.0])),
            PlayerState("def", "Away", np.array([def_x, 0.0]), np.array([0.0, 0.0])),
        ],
        attacking_team_id="Home",
        defending_team_id="Away",
        metadata={"attacking_direction": "left_to_right"},
    )


class TestPPCFPitchControlModel:
    def test_name(self) -> None:
        assert PPCFPitchControlModel().name == "ppcf_pitch_control"

    def test_output_shape_matches_grid(self) -> None:
        grid = _small_grid()
        model = PPCFPitchControlModel()
        surface = model.compute(_two_player_state(), grid)
        assert surface.values.shape == grid.shape

    def test_values_in_unit_interval(self) -> None:
        grid = _small_grid()
        model = PPCFPitchControlModel()
        surface = model.compute(_two_player_state(), grid)
        assert surface.values.min() >= -1e-9
        assert surface.values.max() <= 1.0 + 1e-9

    def test_attacker_controls_nearby_space(self) -> None:
        grid = _small_grid()
        model = PPCFPitchControlModel()
        state = _two_player_state(att_x=40.0, def_x=-40.0)
        surface = model.compute(state, grid)
        mid_col = grid.shape[1] // 2
        assert surface.values[:, mid_col:].mean() > 0.5

    def test_no_attacking_players_raises(self) -> None:
        grid = _small_grid()
        model = PPCFPitchControlModel()
        state = GameState(
            ball_position=np.array([0.0, 0.0]),
            players=[
                PlayerState(
                    "def", "Away", np.array([-10.0, 0.0]), np.array([0.0, 0.0])
                ),
            ],
            attacking_team_id="Home",
            defending_team_id="Away",
        )
        with pytest.raises(ValueError, match="no attacking"):
            model.compute(state, grid)


class TestPPCFParameters:
    def test_default_values(self) -> None:
        p = PPCFParameters()
        assert p.lambda_att == pytest.approx(4.3)
        assert p.kappa_def == pytest.approx(1.0)
        assert p.lambda_def == pytest.approx(4.3)

    def test_time_to_control_properties_positive(self) -> None:
        p = PPCFParameters()
        assert p.time_to_control_att > 0
        assert p.time_to_control_def > 0


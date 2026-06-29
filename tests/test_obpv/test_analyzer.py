"""Tests for obpv.analyzer â€” OBPVAnalyzer."""

from __future__ import annotations

import numpy as np
import pytest

from obpv._grid import PitchGrid
from obpv._state import GameState, PlayerState
from obpv.analyzer import OBPVAnalyzer


def _small_grid() -> PitchGrid:
    return PitchGrid.from_resolution(n_x=6, n_y=4)


def _state(
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    direction: str = "left_to_right",
) -> GameState:
    return GameState(
        ball_position=np.array([ball_x, ball_y]),
        players=[
            PlayerState("att1", "Home", np.array([10.0, 3.0]), np.array([1.0, 0.0])),
            PlayerState("att2", "Home", np.array([15.0, -3.0]), np.array([0.0, 0.0])),
            PlayerState("def1", "Away", np.array([-10.0, 3.0]), np.array([0.0, 0.0])),
            PlayerState("def2", "Away", np.array([-15.0, -3.0]), np.array([0.0, 0.0])),
        ],
        attacking_team_id="Home",
        defending_team_id="Away",
        metadata={"attacking_direction": direction},
    )


class TestOBPVAnalyzer:
    def test_default_builds_without_kernel_csvs(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        assert analyzer is not None

    def test_analyze_returns_obpv_layer(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        result = analyzer.analyze(_state())
        assert "obpv" in result.layers

    def test_component_layers_present(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        result = analyzer.analyze(_state())
        assert "ppcf" in result.layers
        assert "transition" in result.layers
        assert "weight" in result.layers

    def test_obpv_surface_shape_matches_grid(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        result = analyzer.analyze(_state())
        assert result.layers["obpv"].values.shape == grid.shape

    def test_obpv_values_non_negative(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        result = analyzer.analyze(_state())
        assert result.layers["obpv"].values.min() >= -1e-9

    def test_get_layer_raises_on_missing(self) -> None:
        grid = _small_grid()
        analyzer = OBPVAnalyzer.default(grid=grid)
        result = analyzer.analyze(_state())
        with pytest.raises(KeyError):
            result.get_layer("nonexistent")


"""Tests for obpv._grid â€” PitchGrid."""

from __future__ import annotations

import pytest

from obpv._grid import PitchGrid


class TestPitchGrid:
    def test_from_resolution_shape(self) -> None:
        g = PitchGrid.from_resolution(n_x=10, n_y=7)
        assert g.shape == (7, 10)

    def test_x_range_centered(self) -> None:
        g = PitchGrid.from_resolution(length=105.0, n_x=50)
        assert g.x[0] == pytest.approx(-52.5)
        assert g.x[-1] == pytest.approx(52.5)

    def test_y_range_centered(self) -> None:
        g = PitchGrid.from_resolution(width=68.0, n_y=32)
        assert g.y[0] == pytest.approx(-34.0)
        assert g.y[-1] == pytest.approx(34.0)

    def test_meshgrid_returns_two_arrays(self) -> None:
        g = PitchGrid.from_resolution(n_x=5, n_y=4)
        xx, yy = g.meshgrid
        assert xx.shape == (4, 5)
        assert yy.shape == (4, 5)

    def test_goal_positions(self) -> None:
        g = PitchGrid.from_resolution(length=105.0, width=68.0)
        assert g.home_goal == (-52.5, 0)
        assert g.away_goal == (52.5, 0)


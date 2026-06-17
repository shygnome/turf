"""Tests for turf.xt — expected Threat lookup with bilinear interpolation."""

from __future__ import annotations

import pytest

from turf.xt import xt_at, xt_gain


class TestXtAt:
    def test_attacking_penalty_area_higher_than_own_half(self) -> None:
        # penalty spot area (far right, centre) must be far above own half
        assert xt_at(40.0, 0.0) > xt_at(-20.0, 0.0)

    def test_centre_higher_than_flank_at_same_depth(self) -> None:
        # central channel should be more dangerous than the touchline at same x
        assert xt_at(30.0, 0.0) > xt_at(30.0, 30.0)

    def test_all_values_positive(self) -> None:
        for x in (-50.0, -20.0, 0.0, 20.0, 50.0):
            for y in (-30.0, 0.0, 30.0):
                assert xt_at(x, y) > 0

    def test_values_between_zero_and_one(self) -> None:
        for x in (-52.5, 0.0, 52.5):
            for y in (-34.0, 0.0, 34.0):
                v = xt_at(x, y)
                assert 0 < v < 1

    def test_boundary_clamping_does_not_raise(self) -> None:
        # coordinates at and beyond pitch limits should not raise
        assert xt_at(-52.5, -34.0) > 0
        assert xt_at(52.5, 34.0) > 0
        assert xt_at(-60.0, 0.0) > 0   # outside pitch — clamped
        assert xt_at(60.0, 0.0) > 0

    def test_symmetric_about_y_axis(self) -> None:
        # pitch is symmetric: xT(x, y) == xT(x, -y)
        for x in (-30.0, 0.0, 30.0):
            assert xt_at(x, 15.0) == pytest.approx(xt_at(x, -15.0), rel=1e-6)

    def test_bilinear_interpolation_is_smooth(self) -> None:
        # value at a midpoint should be between its two neighbours
        v_left = xt_at(10.0, 0.0)
        v_mid = xt_at(13.28, 0.0)   # midway between two cell centres
        v_right = xt_at(16.56, 0.0)
        assert v_left < v_mid < v_right


class TestXtGain:
    def test_completed_forward_pass_positive_gain(self) -> None:
        # pass from midfield to the box → positive gain
        gain = xt_gain(0.0, 0.0, 40.0, 0.0, completed=True)
        assert gain > 0

    def test_completed_backward_pass_negative_gain(self) -> None:
        # pass from attacking third to own half → negative gain
        gain = xt_gain(30.0, 0.0, -10.0, 0.0, completed=True)
        assert gain < 0

    def test_incomplete_pass_is_zero(self) -> None:
        gain = xt_gain(0.0, 0.0, 40.0, 0.0, completed=False)
        assert gain == pytest.approx(0.0)

    def test_zero_distance_completed_pass_is_zero(self) -> None:
        gain = xt_gain(10.0, 5.0, 10.0, 5.0, completed=True)
        assert gain == pytest.approx(0.0)

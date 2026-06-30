from __future__ import annotations

import pandas as pd

from turf.io.tracking_utils import ball_xy, player_xy

# ---------------------------------------------------------------------------
# player_xy
# ---------------------------------------------------------------------------


class TestPlayerXY:
    def test_valid_positions_passed_through(self) -> None:
        row = pd.Series(
            {
                "Home_1_x": 10.0,
                "Home_1_y": 5.0,
                "Home_2_x": float("nan"),
                "Home_2_y": float("nan"),
            }
        )
        xs, ys = player_xy(row, "Home")
        assert xs == [10.0]
        assert ys == [5.0]

    def test_nan_positions_excluded(self) -> None:
        row = pd.Series(
            {
                "Home_1_x": 10.0,
                "Home_1_y": 5.0,
                "Home_2_x": float("nan"),
                "Home_2_y": float("nan"),
            }
        )
        xs, _ = player_xy(row, "Home")
        assert len(xs) == 1

    def test_away_prefix(self) -> None:
        row = pd.Series({"Away_1_x": -10.0, "Away_1_y": -5.0})
        xs, ys = player_xy(row, "Away")
        assert xs == [-10.0]
        assert ys == [-5.0]

    def test_stops_at_first_missing_column(self) -> None:
        row = pd.Series({"Home_1_x": 2.0, "Home_1_y": 3.0})
        xs, ys = player_xy(row, "Home")
        assert xs == [2.0]
        assert ys == [3.0]

    def test_negative_coords_preserved(self) -> None:
        row = pd.Series({"Home_1_x": -20.0, "Home_1_y": -10.0})
        xs, ys = player_xy(row, "Home")
        assert xs == [-20.0]
        assert ys == [-10.0]


# ---------------------------------------------------------------------------
# ball_xy
# ---------------------------------------------------------------------------


class TestBallXY:
    def test_valid_ball_position_passed_through(self) -> None:
        row = pd.Series({"ball_x": 0.5, "ball_y": 1.0})
        bx, by = ball_xy(row)
        assert bx == 0.5
        assert by == 1.0

    def test_nan_ball_returns_none(self) -> None:
        row = pd.Series({"ball_x": float("nan"), "ball_y": float("nan")})
        bx, by = ball_xy(row)
        assert bx is None
        assert by is None

    def test_missing_ball_column_returns_none(self) -> None:
        row = pd.Series({"Home_1_x": 1.0})
        bx, by = ball_xy(row)
        assert bx is None
        assert by is None

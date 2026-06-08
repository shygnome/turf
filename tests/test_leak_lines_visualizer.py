from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from turf.leak_lines_visualizer import LeakLinesVisualizer

_FPS = 25.0
_META: dict[str, object] = {"period": 1, "event_idx": 0, "team": "Home"}


def _make_defending_frames(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row: dict[str, object] = {
            "Period": 1,
            "Time [s]": float(i),
            "Away_1_x": -10.0 + i * 0.1,
            "Away_1_y": 5.0,
            "Away_1_line": 1.0,
            "Away_2_x": -10.0 + i * 0.1,
            "Away_2_y": -5.0,
            "Away_2_line": 1.0,
            "Away_3_x": -20.0 + i * 0.1,
            "Away_3_y": 0.0,
            "Away_3_line": 2.0,
            "Away_4_x": -20.0 + i * 0.1,
            "Away_4_y": 8.0,
            "Away_4_line": 2.0,
            "Away_5_x": -42.0,
            "Away_5_y": 0.0,
            "Away_5_line": 0.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _make_attacking_frames(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row: dict[str, object] = {
            "Period": 1,
            "Time [s]": float(i),
            "Home_1_x": 10.0 + i * 0.1,
            "Home_1_y": 5.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _players_by_line
# ---------------------------------------------------------------------------


class TestPlayersByLine:
    def test_groups_players_by_line(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                "Away_1_line": 1.0,
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
                "Away_3_x": -20.0,
                "Away_3_y": 0.0,
                "Away_3_line": 2.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert len(result[1]) == 2
        assert len(result[2]) == 1

    def test_nan_x_position_excluded(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": float("nan"),
                "Away_1_y": 5.0,
                "Away_1_line": 1.0,
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert len(result.get(1, [])) == 1

    def test_nan_line_excluded(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                "Away_1_line": float("nan"),
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert len(result.get(1, [])) == 1

    def test_missing_line_col_excluded(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                # Away_1_line intentionally absent
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert len(result.get(1, [])) == 1

    def test_gk_at_line_zero_included(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -42.0,
                "Away_1_y": 0.0,
                "Away_1_line": 0.0,
                "Away_2_x": -10.0,
                "Away_2_y": 5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert 0 in result
        assert len(result[0]) == 1

    def test_coordinates_preserved(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -15.5,
                "Away_1_y": 3.2,
                "Away_1_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._players_by_line(row, "Away")
        assert result[1][0] == (-15.5, 3.2)


# ---------------------------------------------------------------------------
# _player_ids_by_line
# ---------------------------------------------------------------------------


class TestPlayerIdsByLine:
    def test_groups_ids_by_line(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                "Away_1_line": 1.0,
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
                "Away_3_x": -20.0,
                "Away_3_y": 0.0,
                "Away_3_line": 2.0,
            }
        )
        result = LeakLinesVisualizer()._player_ids_by_line(row, "Away")
        assert sorted(result[1]) == ["1", "2"]
        assert result[2] == ["3"]

    def test_nan_x_excludes_player(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": float("nan"),
                "Away_1_y": 5.0,
                "Away_1_line": 1.0,
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._player_ids_by_line(row, "Away")
        assert result.get(1, []) == ["2"]

    def test_nan_line_excludes_player(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                "Away_1_line": float("nan"),
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._player_ids_by_line(row, "Away")
        assert result.get(1, []) == ["2"]

    def test_missing_line_col_excludes_player(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": -10.0,
                "Away_1_y": 5.0,
                # Away_1_line intentionally absent
                "Away_2_x": -10.0,
                "Away_2_y": -5.0,
                "Away_2_line": 1.0,
            }
        )
        result = LeakLinesVisualizer()._player_ids_by_line(row, "Away")
        assert result.get(1, []) == ["2"]


# ---------------------------------------------------------------------------
# _repr_x
# ---------------------------------------------------------------------------


class TestReprX:
    def test_picks_deepest_toward_positive_gk(self) -> None:
        # GK at +42; deepest = highest x = 14.0
        players = [(10.0, 5.0), (14.0, -5.0)]
        assert LeakLinesVisualizer()._repr_x(players, gk_x=42.0) == pytest.approx(14.0)

    def test_picks_deepest_toward_negative_gk(self) -> None:
        # GK at -42; deepest = most negative x = -14.0
        players = [(-10.0, 5.0), (-14.0, -5.0)]
        result = LeakLinesVisualizer()._repr_x(players, gk_x=-42.0)
        assert result == pytest.approx(-14.0)

    def test_single_player_returns_its_x(self) -> None:
        result = LeakLinesVisualizer()._repr_x([(7.5, 3.0)], gk_x=42.0)
        assert result == pytest.approx(7.5)

    def test_ignores_player_on_wrong_side_of_center(self) -> None:
        # GK at +42; one player crossed to -5 (wrong side), one at +3
        # Should pick +3 (closest to positive GK), not -5
        players = [(-5.0, 3.0), (3.0, 0.0)]
        assert LeakLinesVisualizer()._repr_x(players, gk_x=42.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# smooth_lines parameter
# ---------------------------------------------------------------------------


class TestSmoothLinesParam:
    def test_smooth_lines_true_calls_smooth_line_assignments(self) -> None:
        from unittest.mock import patch

        with patch(
            "turf.leak_lines_visualizer.smooth_line_assignments"
        ) as mock_smooth:
            mock_smooth.side_effect = lambda df, team, **kw: df
            LeakLinesVisualizer().animate(
                _make_defending_frames(),
                _make_attacking_frames(),
                "Away",
                "Home",
                _META,
                smooth_lines=True,
            )
            assert mock_smooth.called
        plt.close("all")

    def test_smooth_lines_false_skips_smooth_line_assignments(self) -> None:
        from unittest.mock import patch

        with patch(
            "turf.leak_lines_visualizer.smooth_line_assignments"
        ) as mock_smooth:
            mock_smooth.side_effect = lambda df, team, **kw: df
            LeakLinesVisualizer().animate(
                _make_defending_frames(),
                _make_attacking_frames(),
                "Away",
                "Home",
                _META,
                smooth_lines=False,
            )
            assert not mock_smooth.called
        plt.close("all")


# ---------------------------------------------------------------------------
# animate
# ---------------------------------------------------------------------------


class TestAnimate:
    def test_returns_func_animation(self) -> None:
        from matplotlib.animation import FuncAnimation

        anim = LeakLinesVisualizer().animate(
            _make_defending_frames(),
            _make_attacking_frames(),
            "Away",
            "Home",
            _META,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_frame_count_matches_defending_frames(self) -> None:
        with patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA:
            MockFA.return_value = MagicMock()
            LeakLinesVisualizer().animate(
                _make_defending_frames(n=7),
                _make_attacking_frames(n=7),
                "Away",
                "Home",
                _META,
            )
            call_kwargs = MockFA.call_args
            frames_arg = call_kwargs.kwargs.get("frames") or call_kwargs.args[2]
            assert frames_arg == 7

    def test_update_fn_does_not_raise_with_valid_frame(self) -> None:
        captured_update: list[object] = []

        with (
            patch("turf.leak_lines_visualizer.Pitch") as MockPitch,
            patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA,
        ):

            def _capture(_fig: object, func: object, **_kw: object) -> MagicMock:
                captured_update.append(func)
                return MagicMock()

            MockFA.side_effect = _capture
            mock_pitch_instance = MagicMock()
            MockPitch.return_value = mock_pitch_instance
            fig, real_ax = plt.subplots()
            mock_pitch_instance.draw.return_value = (fig, real_ax)

            LeakLinesVisualizer().animate(
                _make_defending_frames(n=3),
                _make_attacking_frames(n=3),
                "Away",
                "Home",
                _META,
            )
            update_fn = captured_update[0]
            assert callable(update_fn)
            result = update_fn(0)  # type: ignore[operator]
            assert result is not None
            plt.close(fig)

    def test_update_fn_clears_artists_when_all_players_absent(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "Period": 1,
                "Time [s]": 0.0,
                "Away_1_x": float("nan"),
                "Away_1_y": float("nan"),
                "Away_1_line": float("nan"),
                "ball_x": float("nan"),
                "ball_y": float("nan"),
            }
        ]
        def_frames = pd.DataFrame(rows)
        atk_frames = _make_attacking_frames(n=1)

        captured_update: list[object] = []
        with (
            patch("turf.leak_lines_visualizer.Pitch") as MockPitch,
            patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA,
        ):

            def _capture(_fig: object, func: object, **_kw: object) -> MagicMock:
                captured_update.append(func)
                return MagicMock()

            MockFA.side_effect = _capture
            mock_pitch_instance = MagicMock()
            MockPitch.return_value = mock_pitch_instance
            fig, real_ax = plt.subplots()
            mock_pitch_instance.draw.return_value = (fig, real_ax)

            LeakLinesVisualizer().animate(
                def_frames, atk_frames, "Away", "Home", _META
            )
            update_fn = captured_update[0]
            update_fn(0)  # type: ignore[operator]  # should not raise
            plt.close(fig)

    def test_debug_true_does_not_raise(self) -> None:
        from matplotlib.animation import FuncAnimation

        anim = LeakLinesVisualizer().animate(
            _make_defending_frames(),
            _make_attacking_frames(),
            "Away",
            "Home",
            _META,
            debug=True,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_animate_home_defending_away_attacking(self) -> None:
        from matplotlib.animation import FuncAnimation

        # Swap: Home defends, Away attacks
        def_frames = pd.DataFrame(
            [
                {
                    "Period": 1,
                    "Time [s]": 0.0,
                    "Home_1_x": -10.0,
                    "Home_1_y": 5.0,
                    "Home_1_line": 1.0,
                    "Home_2_x": -10.0,
                    "Home_2_y": -5.0,
                    "Home_2_line": 1.0,
                    "Home_3_x": -42.0,
                    "Home_3_y": 0.0,
                    "Home_3_line": 0.0,
                    "ball_x": 0.0,
                    "ball_y": 0.0,
                }
            ]
        )
        atk_frames = pd.DataFrame(
            [
                {
                    "Period": 1,
                    "Time [s]": 0.0,
                    "Away_1_x": 10.0,
                    "Away_1_y": 3.0,
                    "ball_x": 0.0,
                    "ball_y": 0.0,
                }
            ]
        )
        anim = LeakLinesVisualizer().animate(
            def_frames,
            atk_frames,
            "Home",
            "Away",
            {"period": 1, "event_idx": 0, "team": "Away"},
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


# ---------------------------------------------------------------------------
# _find_player_gk
# ---------------------------------------------------------------------------


class TestFindPlayerGk:
    def test_returns_id_of_max_abs_x_positive(self) -> None:
        row = pd.Series(
            {"Away_1_x": 10.0, "Away_2_x": 43.0, "Away_3_x": 5.0}
        )
        assert LeakLinesVisualizer()._find_player_gk(row, "Away") == "2"

    def test_returns_id_of_max_abs_x_negative(self) -> None:
        row = pd.Series(
            {"Home_1_x": -10.0, "Home_2_x": -43.0, "Home_3_x": -5.0}
        )
        assert LeakLinesVisualizer()._find_player_gk(row, "Home") == "2"

    def test_returns_none_when_no_columns(self) -> None:
        row = pd.Series({"ball_x": 0.0, "ball_y": 0.0})
        assert LeakLinesVisualizer()._find_player_gk(row, "Away") is None

    def test_nan_x_excluded(self) -> None:
        row = pd.Series(
            {"Away_1_x": float("nan"), "Away_2_x": 30.0}
        )
        assert LeakLinesVisualizer()._find_player_gk(row, "Away") == "2"


# ---------------------------------------------------------------------------
# _ball_line_label
# ---------------------------------------------------------------------------


class TestBallLineLabel:
    def test_empty_dict_returns_empty_string(self) -> None:
        assert LeakLinesVisualizer()._ball_line_label(10.0, {}) == ""

    def test_ball_not_past_any_line(self) -> None:
        # Ball at x=5, line 1 repr at x=9 — ball hasn't beaten any line
        assert LeakLinesVisualizer()._ball_line_label(5.0, {1: 9.0}) == ""

    def test_ball_past_first_line_only(self) -> None:
        # Only the most-advanced line beaten → L2 (least dangerous in user conv.)
        label = LeakLinesVisualizer()._ball_line_label(
            12.0, {1: 9.0, 2: 19.0}
        )
        assert label == "L2"

    def test_ball_past_two_lines(self) -> None:
        label = LeakLinesVisualizer()._ball_line_label(
            22.0, {1: 9.0, 2: 19.0, 3: 26.0}
        )
        assert label == "L2"

    def test_ball_past_all_lines(self) -> None:
        # Ball behind deepest line = L1 in user convention (most dangerous)
        label = LeakLinesVisualizer()._ball_line_label(
            30.0, {1: 9.0, 2: 19.0, 3: 26.0}
        )
        assert label == "L1"

    def test_ball_on_opposite_side_returns_empty(self) -> None:
        # Defending at positive x, ball at negative x — hasn't beaten any line
        assert LeakLinesVisualizer()._ball_line_label(-5.0, {1: 9.0, 2: 19.0}) == ""

    def test_ball_negative_side_defending(self) -> None:
        # Defending at negative x, ball at x=-22 has beaten L2 (repr=-19)
        label = LeakLinesVisualizer()._ball_line_label(
            -22.0, {1: -9.0, 2: -19.0, 3: -26.0}
        )
        assert label == "L2"


# ---------------------------------------------------------------------------
# _player_xy_with_ids
# ---------------------------------------------------------------------------


class TestPlayerXyWithIds:
    def test_returns_triples_with_player_id(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": 10.0,
                "Away_1_y": 5.0,
                "Away_2_x": 20.0,
                "Away_2_y": -5.0,
            }
        )
        result = LeakLinesVisualizer()._player_xy_with_ids(row, "Away")
        assert ("1", 10.0, 5.0) in result
        assert ("2", 20.0, -5.0) in result

    def test_nan_x_excluded(self) -> None:
        row = pd.Series(
            {
                "Away_1_x": float("nan"),
                "Away_1_y": 5.0,
                "Away_2_x": 20.0,
                "Away_2_y": -5.0,
            }
        )
        result = LeakLinesVisualizer()._player_xy_with_ids(row, "Away")
        assert len(result) == 1
        assert result[0][0] == "2"

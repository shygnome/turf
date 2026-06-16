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


def _make_two_line_away_frames(n: int = 5) -> pd.DataFrame:
    """Away defending (GK at -42). Line 1 at x=-5 (front), line 2 at x=-20 (deep)."""
    rows = []
    for i in range(n):
        row: dict[str, object] = {
            "Period": 1,
            "Time [s]": float(i),
            "Away_1_x": -5.0,
            "Away_1_y": 5.0,
            "Away_1_line": 1.0,
            "Away_2_x": -5.0,
            "Away_2_y": -5.0,
            "Away_2_line": 1.0,
            "Away_3_x": -20.0,
            "Away_3_y": 5.0,
            "Away_3_line": 2.0,
            "Away_4_x": -20.0,
            "Away_4_y": -5.0,
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

        with patch("turf.leak_lines_visualizer.smooth_line_assignments") as mock_smooth:
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

        with patch("turf.leak_lines_visualizer.smooth_line_assignments") as mock_smooth:
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

            LeakLinesVisualizer().animate(def_frames, atk_frames, "Away", "Home", _META)
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
        row = pd.Series({"Away_1_x": 10.0, "Away_2_x": 43.0, "Away_3_x": 5.0})
        assert LeakLinesVisualizer()._find_player_gk(row, "Away") == "2"

    def test_returns_id_of_max_abs_x_negative(self) -> None:
        row = pd.Series({"Home_1_x": -10.0, "Home_2_x": -43.0, "Home_3_x": -5.0})
        assert LeakLinesVisualizer()._find_player_gk(row, "Home") == "2"

    def test_returns_none_when_no_columns(self) -> None:
        row = pd.Series({"ball_x": 0.0, "ball_y": 0.0})
        assert LeakLinesVisualizer()._find_player_gk(row, "Away") is None

    def test_nan_x_excluded(self) -> None:
        row = pd.Series({"Away_1_x": float("nan"), "Away_2_x": 30.0})
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
        # Only the most-advanced line beaten → L1 (front line in new convention)
        label = LeakLinesVisualizer()._ball_line_label(12.0, {1: 9.0, 2: 19.0})
        assert label == "L1"

    def test_ball_past_two_lines(self) -> None:
        label = LeakLinesVisualizer()._ball_line_label(22.0, {1: 9.0, 2: 19.0, 3: 26.0})
        assert label == "L2"

    def test_ball_past_all_lines(self) -> None:
        # Ball past all 3 lines → L3 (deepest line in new L1=front convention)
        label = LeakLinesVisualizer()._ball_line_label(30.0, {1: 9.0, 2: 19.0, 3: 26.0})
        assert label == "L3"

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


# ---------------------------------------------------------------------------
# _closest_player_xy
# ---------------------------------------------------------------------------


class TestClosestPlayerXy:
    def _row(self, players: list[tuple[float, float]], team: str = "Home") -> pd.Series:
        d: dict[str, object] = {"ball_x": 0.0, "ball_y": 0.0}
        for i, (x, y) in enumerate(players, start=1):
            d[f"{team}_{i}_x"] = x
            d[f"{team}_{i}_y"] = y
        return pd.Series(d)

    def test_returns_closest_player(self) -> None:
        row = self._row([(10.0, 0.0), (2.0, 0.0), (20.0, 0.0)])
        result = LeakLinesVisualizer()._closest_player_xy(row, "Home", 0.0, 0.0)
        assert result == pytest.approx((2.0, 0.0))

    def test_returns_none_when_no_players(self) -> None:
        row = pd.Series({"ball_x": 0.0, "ball_y": 0.0})
        result = LeakLinesVisualizer()._closest_player_xy(row, "Home", 0.0, 0.0)
        assert result is None

    def test_nan_position_excluded(self) -> None:
        row = self._row([(float("nan"), 0.0), (5.0, 0.0)])
        result = LeakLinesVisualizer()._closest_player_xy(row, "Home", 0.0, 0.0)
        assert result == pytest.approx((5.0, 0.0))

    def test_single_player(self) -> None:
        row = self._row([(7.0, 3.0)])
        result = LeakLinesVisualizer()._closest_player_xy(row, "Home", 0.0, 0.0)
        assert result == pytest.approx((7.0, 3.0))


# ---------------------------------------------------------------------------
# animate with pass_labels (reveal animation)
# ---------------------------------------------------------------------------


def _make_pass_labels(
    hull_vertices: list[list[tuple[float, float]]] | None = None,
    is_line_breaking: bool = True,
    lines_broken: list[int] | None = None,
    direction_per_line: list[str] | None = None,
    location_after_break: str | None = "Inside",
) -> dict[str, object]:
    return {
        "is_line_breaking": is_line_breaking,
        "lines_broken_count": len(lines_broken or [1]),
        "lines_broken": lines_broken if lines_broken is not None else [1],
        "direction_per_line": (
            direction_per_line if direction_per_line is not None else ["Through"]
        ),
        "location_after_break": location_after_break,
        "hull_vertices": (
            hull_vertices
            if hull_vertices is not None
            else [[(0.0, -5.0), (0.0, 5.0), (10.0, 5.0), (10.0, -5.0)]]
        ),
    }


class TestAnimateWithLabels:
    _FPS = 10.0
    _FREEZE = 5.0
    _N = 5

    def _run_animate_capture(
        self,
        pass_labels: dict[str, object] | None,
        freeze_duration: float = _FREEZE,
        n: int = _N,
    ) -> tuple[object, object, object]:
        """Run animate() and return (update_fn, real_ax, fig)."""
        captured: list[object] = []
        with (
            patch("turf.leak_lines_visualizer.Pitch") as MockPitch,
            patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA,
        ):

            def _capture(_fig: object, func: object, **_kw: object) -> MagicMock:
                captured.append(func)
                return MagicMock()

            MockFA.side_effect = _capture
            mock_pitch_instance = MagicMock()
            MockPitch.return_value = mock_pitch_instance
            fig, real_ax = plt.subplots()
            mock_pitch_instance.draw.return_value = (fig, real_ax)

            LeakLinesVisualizer().animate(
                _make_defending_frames(n=n),
                _make_attacking_frames(n=n),
                "Away",
                "Home",
                _META,
                fps=self._FPS,
                pass_labels=pass_labels,
                freeze_duration=freeze_duration,
            )
        return captured[0], real_ax, fig

    def test_frame_count_includes_freeze(self) -> None:
        with patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA:
            MockFA.return_value = MagicMock()
            LeakLinesVisualizer().animate(
                _make_defending_frames(n=self._N),
                _make_attacking_frames(n=self._N),
                "Away",
                "Home",
                _META,
                fps=self._FPS,
                pass_labels=_make_pass_labels(),
                freeze_duration=self._FREEZE,
            )
            call_kwargs = MockFA.call_args
            frames_arg = call_kwargs.kwargs.get("frames") or call_kwargs.args[2]
            assert frames_arg == self._N + int(self._FREEZE * self._FPS)

    def test_frame_count_no_labels_unchanged(self) -> None:
        with patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA:
            MockFA.return_value = MagicMock()
            LeakLinesVisualizer().animate(
                _make_defending_frames(n=self._N),
                _make_attacking_frames(n=self._N),
                "Away",
                "Home",
                _META,
                fps=self._FPS,
                pass_labels=None,
            )
            call_kwargs = MockFA.call_args
            frames_arg = call_kwargs.kwargs.get("frames") or call_kwargs.args[2]
            assert frames_arg == self._N

    def test_hull_patch_not_present_when_no_labels(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        _update, real_ax, fig = self._run_animate_capture(pass_labels=None)
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert len(hulls) == 0
        plt.close(fig)

    def test_hull_patch_exists_when_labels_provided(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        _update, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert len(hulls) == 1
        plt.close(fig)

    def test_hull_not_visible_at_last_tracking_frame(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N - 1)  # type: ignore[operator]
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert not hulls[0].get_visible()
        plt.close(fig)

    def test_hull_visible_after_phase1(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N + int(self._FPS * 1))  # type: ignore[operator]
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert hulls[0].get_visible()
        plt.close(fig)

    def test_pass_arrow_not_present_when_no_labels(self) -> None:
        from matplotlib.patches import FancyArrowPatch

        _update, real_ax, fig = self._run_animate_capture(pass_labels=None)
        arrows = [p for p in real_ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 0
        plt.close(fig)

    def test_pass_arrow_not_visible_at_last_tracking_frame(self) -> None:
        from matplotlib.patches import FancyArrowPatch

        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N - 1)  # type: ignore[operator]
        arrows = [p for p in real_ax.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == 1
        assert not arrows[0].get_visible()
        plt.close(fig)

    def test_pass_arrow_visible_after_phase2(self) -> None:
        from matplotlib.patches import FancyArrowPatch

        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N + int(self._FPS * 2))  # type: ignore[operator]
        arrows = [p for p in real_ax.patches if isinstance(p, FancyArrowPatch)]
        assert arrows[0].get_visible()
        plt.close(fig)

    def test_info_text_not_visible_before_phase3(self) -> None:
        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N + int(self._FPS * 2))  # type: ignore[operator]
        info_texts = [
            t for t in real_ax.texts if t.get_bbox_patch() is not None
        ]
        assert all(not t.get_visible() for t in info_texts)
        plt.close(fig)

    def test_info_text_visible_after_phase3(self) -> None:
        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        update_fn(self._N + int(self._FPS * 3))  # type: ignore[operator]
        info_texts = [
            t
            for t in real_ax.texts
            if t.get_bbox_patch() is not None and t.get_visible()
        ]
        assert len(info_texts) == 1
        assert info_texts[0].get_text() != ""
        plt.close(fig)

    def test_update_does_not_raise_across_all_reveal_frames(self) -> None:
        n = self._N
        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        total = n + int(self._FREEZE * self._FPS)
        for i in range(total):
            update_fn(i)  # type: ignore[operator]
        plt.close(fig)

    def test_hull_patch_uses_hatch_pattern(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        _update, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert len(hulls) == 1
        assert hulls[0].get_hatch() is not None and hulls[0].get_hatch() != ""
        plt.close(fig)

    def test_hull_patch_no_solid_fill(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        _update, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        # fill should be False (no solid colour background)
        assert not hulls[0].get_fill()
        plt.close(fig)

    def test_two_hulls_when_two_provided(self) -> None:
        from matplotlib.patches import Polygon as MplPolygon

        two_hull_labels = _make_pass_labels(
            hull_vertices=[
                [(0.0, -5.0), (0.0, 5.0), (5.0, 5.0), (5.0, -5.0)],
                [(5.0, -5.0), (5.0, 5.0), (10.0, 5.0), (10.0, -5.0)],
            ]
        )
        _update, real_ax, fig = self._run_animate_capture(pass_labels=two_hull_labels)
        hulls = [p for p in real_ax.patches if isinstance(p, MplPolygon)]
        assert len(hulls) == 2
        plt.close(fig)

    def test_line_labels_hidden_during_reveal(self) -> None:
        update_fn, real_ax, fig = self._run_animate_capture(
            pass_labels=_make_pass_labels()
        )
        # Prime a tracking frame so labels become visible
        update_fn(0)  # type: ignore[operator]
        # Enter reveal phase (first freeze frame)
        update_fn(self._N)  # type: ignore[operator]
        # Unit-line text labels (no bbox, at bottom of pitch) should be hidden
        label_texts = [
            t
            for t in real_ax.texts
            if t.get_bbox_patch() is None and t.get_visible()
            and t.get_text().startswith("L")
        ]
        assert len(label_texts) == 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# L-label ordering convention: L1 = front (most advanced), L4 = deepest
# ---------------------------------------------------------------------------


class TestLabelOrderingConvention:
    """L1 must label the most-advanced (front/smallest |repr_x|) unit line."""

    _FPS = 10.0
    _N = 5

    def _capture_animate(
        self, def_frames: pd.DataFrame, atk_frames: pd.DataFrame
    ) -> tuple[object, object, object]:
        captured: list[object] = []
        with (
            patch("turf.leak_lines_visualizer.Pitch") as MockPitch,
            patch("turf.leak_lines_visualizer.FuncAnimation") as MockFA,
        ):

            def _cap(_fig: object, func: object, **_kw: object) -> MagicMock:
                captured.append(func)
                return MagicMock()

            MockFA.side_effect = _cap
            mock_pitch_instance = MagicMock()
            MockPitch.return_value = mock_pitch_instance
            fig, real_ax = plt.subplots()
            mock_pitch_instance.draw.return_value = (fig, real_ax)
            LeakLinesVisualizer().animate(
                def_frames,
                atk_frames,
                "Away",
                "Home",
                _META,
                fps=self._FPS,
                smooth_lines=False,
            )
        return captured[0], real_ax, fig

    def _visible_label_x(self, ax: object) -> dict[str, float]:
        import matplotlib.axes as _mpl_axes

        assert isinstance(ax, _mpl_axes.Axes)
        return {
            t.get_text(): t.get_position()[0]
            for t in ax.texts
            if t.get_text().startswith("L") and t.get_visible()
        }

    def test_front_line_is_l1(self) -> None:
        """Line closest to centre (smallest |repr_x|) must be labelled L1."""
        update_fn, real_ax, fig = self._capture_animate(
            _make_two_line_away_frames(self._N), _make_attacking_frames(self._N)
        )
        update_fn(0)  # type: ignore[operator]
        pos = self._visible_label_x(real_ax)
        assert "L1" in pos and "L2" in pos
        assert abs(pos["L1"]) < abs(pos["L2"])
        plt.close(fig)

    def test_deep_line_is_l2(self) -> None:
        """Line farthest from centre (largest |repr_x|) must be labelled L2."""
        update_fn, real_ax, fig = self._capture_animate(
            _make_two_line_away_frames(self._N), _make_attacking_frames(self._N)
        )
        update_fn(0)  # type: ignore[operator]
        pos = self._visible_label_x(real_ax)
        assert abs(pos["L2"]) > abs(pos["L1"])
        plt.close(fig)

    def test_label_order_stable_across_frames(self) -> None:
        """The L1/L2 assignment must not flip between the first and last frame."""
        update_fn, real_ax, fig = self._capture_animate(
            _make_two_line_away_frames(self._N), _make_attacking_frames(self._N)
        )

        def _positions() -> dict[str, float]:
            return self._visible_label_x(real_ax)

        update_fn(0)  # type: ignore[operator]
        first = _positions()

        update_fn(self._N - 1)  # type: ignore[operator]
        last = _positions()

        assert set(first) == set(last)
        for label, x in first.items():
            assert x == pytest.approx(last[label])
        plt.close(fig)

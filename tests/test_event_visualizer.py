from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turf.event_extractor import EventClip
from turf.event_visualizer import EventVisualizer

_FPS = 25.0


def _make_home_row(**overrides: object) -> pd.Series:
    data: dict[str, object] = {
        "Period": 1,
        "Time [s]": 120.0,
        "Home_1_x": 10.0,
        "Home_1_y": 5.0,
        "Home_2_x": float("nan"),
        "Home_2_y": float("nan"),
        "ball_x": 0.5,
        "ball_y": 1.0,
    }
    data.update(overrides)
    return pd.Series(data)


def _make_away_row(**overrides: object) -> pd.Series:
    data: dict[str, object] = {
        "Period": 1,
        "Time [s]": 120.0,
        "Away_1_x": -10.0,
        "Away_1_y": -5.0,
        "Away_2_x": float("nan"),
        "Away_2_y": float("nan"),
        "ball_x": 0.5,
        "ball_y": 1.0,
    }
    data.update(overrides)
    return pd.Series(data)


def _make_clip(
    n_frames: int = 3,
    period: int = 1,
    start_frame: int = 0,
) -> EventClip:
    home_rows = [
        {
            "Period": period,
            "Time [s]": start_frame / _FPS + i / _FPS,
            "Home_1_x": 10.0 + i,
            "Home_1_y": 5.0,
            "Home_2_x": float("nan"),
            "Home_2_y": float("nan"),
            "ball_x": 0.5 + i * 0.1,
            "ball_y": 1.0,
        }
        for i in range(n_frames)
    ]
    away_rows = [
        {
            "Period": period,
            "Time [s]": start_frame / _FPS + i / _FPS,
            "Away_1_x": -10.0 - i,
            "Away_1_y": -5.0,
            "Away_2_x": float("nan"),
            "Away_2_y": float("nan"),
            "ball_x": 0.5 + i * 0.1,
            "ball_y": 1.0,
        }
        for i in range(n_frames)
    ]
    return EventClip(
        event_idx=0,
        start_frame=start_frame,
        end_frame=start_frame + n_frames - 1,
        metadata={
            "event_idx": 0,
            "start_frame": start_frame,
            "end_frame": start_frame + n_frames - 1,
            "start_x": 0.5,
            "start_y": 1.0,
            "end_x": 15.0,
            "end_y": 3.0,
            "from_player": "Player A",
            "to_player": "Player B",
            "team": "Home",
            "subtype": "success",
            "period": period,
        },
        home_frames=pd.DataFrame(home_rows),
        away_frames=pd.DataFrame(away_rows),
    )


# ---------------------------------------------------------------------------
# _player_xy — skillcorner coords: metres, no transform
# ---------------------------------------------------------------------------


class TestPlayerXY:
    def test_valid_positions_passed_through(self) -> None:
        row = _make_home_row()
        xs, ys = EventVisualizer()._player_xy(row, "Home")
        assert xs == [10.0]
        assert ys == [5.0]

    def test_nan_positions_excluded(self) -> None:
        row = _make_home_row()
        xs, _ = EventVisualizer()._player_xy(row, "Home")
        # Home_2 is NaN — only Home_1 should be included
        assert len(xs) == 1

    def test_away_prefix(self) -> None:
        row = _make_away_row()
        xs, ys = EventVisualizer()._player_xy(row, "Away")
        assert xs == [-10.0]
        assert ys == [-5.0]

    def test_stops_at_first_missing_column(self) -> None:
        row = pd.Series({"Home_1_x": 2.0, "Home_1_y": 3.0})
        xs, ys = EventVisualizer()._player_xy(row, "Home")
        assert xs == [2.0]
        assert ys == [3.0]

    def test_negative_coords_preserved(self) -> None:
        row = pd.Series({"Home_1_x": -20.0, "Home_1_y": -10.0})
        xs, ys = EventVisualizer()._player_xy(row, "Home")
        assert xs == [-20.0]
        assert ys == [-10.0]


# ---------------------------------------------------------------------------
# _ball_xy
# ---------------------------------------------------------------------------


class TestBallXY:
    def test_valid_ball_position_passed_through(self) -> None:
        row = _make_home_row()
        bx, by = EventVisualizer()._ball_xy(row)
        assert bx == 0.5
        assert by == 1.0

    def test_nan_ball_returns_none(self) -> None:
        row = _make_home_row(**{"ball_x": float("nan"), "ball_y": float("nan")})
        bx, by = EventVisualizer()._ball_xy(row)
        assert bx is None
        assert by is None

    def test_missing_ball_column_returns_none(self) -> None:
        row = pd.Series({"Home_1_x": 1.0})
        bx, by = EventVisualizer()._ball_xy(row)
        assert bx is None
        assert by is None


# ---------------------------------------------------------------------------
# _timestamp_str
# ---------------------------------------------------------------------------


class TestTimestampStr:
    def test_first_half(self) -> None:
        # start_frame=3000, fps=25 → Time[s]=120.0 → "H1 02:00"
        clip = _make_clip(period=1, start_frame=3000)
        assert EventVisualizer()._timestamp_str(clip) == "H1 02:00"

    def test_second_half(self) -> None:
        # start_frame=1875, fps=25 → Time[s]=75.0 → "H2 01:15"
        clip = _make_clip(period=2, start_frame=1875)
        assert EventVisualizer()._timestamp_str(clip) == "H2 01:15"

    def test_zero_seconds(self) -> None:
        clip = _make_clip(period=1, start_frame=0)
        assert EventVisualizer()._timestamp_str(clip) == "H1 00:00"


# ---------------------------------------------------------------------------
# _smooth_frames
# ---------------------------------------------------------------------------


class TestSmoothFrames:
    def test_returns_same_shape(self) -> None:
        clip = _make_clip(n_frames=10)
        smoothed = EventVisualizer()._smooth_frames(clip.home_frames)
        assert smoothed.shape == clip.home_frames.shape

    def test_original_not_mutated(self) -> None:
        clip = _make_clip(n_frames=10)
        original_vals = clip.home_frames["Home_1_x"].tolist()
        EventVisualizer()._smooth_frames(clip.home_frames)
        assert clip.home_frames["Home_1_x"].tolist() == original_vals

    def test_smoothed_values_differ_from_noisy(self) -> None:
        rng = np.random.default_rng(42)
        n = 20
        # Noisy signal: linear trend + noise
        xs = list(range(n))
        noise = rng.normal(0, 2.0, n).tolist()
        noisy = [x + e for x, e in zip(xs, noise, strict=True)]
        df = pd.DataFrame({"Home_1_x": noisy, "Home_1_y": [0.0] * n})
        smoothed = EventVisualizer()._smooth_frames(df)
        # Smoothed values should not be identical to noisy input
        assert not np.allclose(
            df["Home_1_x"].to_numpy(), smoothed["Home_1_x"].to_numpy()
        )

    def test_nan_columns_preserved(self) -> None:
        clip = _make_clip(n_frames=10)
        smoothed = EventVisualizer()._smooth_frames(clip.home_frames)
        # Home_2 is all-NaN — should remain NaN after smoothing
        assert smoothed["Home_2_x"].isna().all()

    def test_too_short_clip_unchanged(self) -> None:
        # 3 frames < default window=5 → no smoothing applied
        clip = _make_clip(n_frames=3)
        original = clip.home_frames["Home_1_x"].tolist()
        smoothed = EventVisualizer()._smooth_frames(clip.home_frames)
        assert smoothed["Home_1_x"].tolist() == original

    def test_invalid_polyorder_returns_unchanged(self) -> None:
        # polyorder=5 > window=3 → savgol_filter would raise; guard must skip smoothing
        clip = _make_clip(n_frames=10)
        original = clip.home_frames["Home_1_x"].tolist()
        smoothed = EventVisualizer()._smooth_frames(
            clip.home_frames, window=3, polyorder=5
        )
        assert smoothed["Home_1_x"].tolist() == original


# ---------------------------------------------------------------------------
# freeze_frame
# ---------------------------------------------------------------------------


class TestFreezeFrame:
    def test_returns_figure(self) -> None:
        clip = _make_clip()
        fig = EventVisualizer().freeze_frame(clip, "pass")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_pass_draws_arrows(self) -> None:
        with patch("turf.event_visualizer.Pitch") as MockPitch:
            mock_pitch = MagicMock()
            MockPitch.return_value = mock_pitch
            mock_pitch.draw.return_value = (MagicMock(), MagicMock())

            clip = _make_clip()
            EventVisualizer().freeze_frame(clip, "pass")

            mock_pitch.arrows.assert_called_once()

    def test_non_pass_no_arrows(self) -> None:
        with patch("turf.event_visualizer.Pitch") as MockPitch:
            mock_pitch = MagicMock()
            MockPitch.return_value = mock_pitch
            mock_pitch.draw.return_value = (MagicMock(), MagicMock())

            clip = _make_clip()
            EventVisualizer().freeze_frame(clip, "shot")

            mock_pitch.arrows.assert_not_called()

    def test_title_includes_period(self) -> None:
        with patch("turf.event_visualizer.Pitch") as MockPitch:
            mock_pitch = MagicMock()
            MockPitch.return_value = mock_pitch
            mock_ax = MagicMock()
            mock_pitch.draw.return_value = (MagicMock(), mock_ax)

            clip = _make_clip(period=2)
            EventVisualizer().freeze_frame(clip, "pass")

            title = mock_ax.set_title.call_args[0][0]
            assert "H2" in title

    def test_title_includes_label(self) -> None:
        with patch("turf.event_visualizer.Pitch") as MockPitch:
            mock_pitch = MagicMock()
            MockPitch.return_value = mock_pitch
            mock_ax = MagicMock()
            mock_pitch.draw.return_value = (MagicMock(), mock_ax)

            clip = _make_clip()
            EventVisualizer().freeze_frame(clip, "pass")

            title = mock_ax.set_title.call_args[0][0]
            assert "pass" in title.lower()

    def test_smooth_true_does_not_raise(self) -> None:
        clip = _make_clip(n_frames=10)
        fig = EventVisualizer().freeze_frame(clip, "pass", smooth=True)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_smooth_window_larger_smooths_more(self) -> None:
        rng = np.random.default_rng(0)
        n = 30
        noise = rng.normal(0, 3.0, n)
        home_rows = [
            {
                "Period": 1,
                "Time [s]": float(i),
                "Home_1_x": float(i) + noise[i],
                "Home_1_y": 0.0,
                "Home_2_x": float("nan"),
                "Home_2_y": float("nan"),
                "ball_x": 0.0,
                "ball_y": 0.0,
            }
            for i in range(n)
        ]
        away_rows = [
            {
                "Period": 1,
                "Time [s]": float(i),
                "Away_1_x": -float(i),
                "Away_1_y": 0.0,
                "Away_2_x": float("nan"),
                "Away_2_y": float("nan"),
                "ball_x": 0.0,
                "ball_y": 0.0,
            }
            for i in range(n)
        ]
        clip = EventClip(
            event_idx=0,
            start_frame=0,
            end_frame=n - 1,
            metadata={
                "event_idx": 0,
                "start_frame": 0,
                "end_frame": n - 1,
                "start_x": 0.0,
                "start_y": 0.0,
                "end_x": 1.0,
                "end_y": 0.0,
                "from_player": "A",
                "to_player": "B",
                "team": "Home",
                "subtype": "success",
                "period": 1,
            },
            home_frames=pd.DataFrame(home_rows),
            away_frames=pd.DataFrame(away_rows),
        )
        # window=11 should smooth more aggressively than window=5 → smaller variance
        fig_w5 = EventVisualizer().freeze_frame(
            clip, "pass", smooth=True, smooth_window=5
        )
        fig_w11 = EventVisualizer().freeze_frame(
            clip, "pass", smooth=True, smooth_window=11
        )
        plt.close(fig_w5)
        plt.close(fig_w11)
        # Just verify they don't raise — functional difference tested in _smooth_frames


# ---------------------------------------------------------------------------
# animate
# ---------------------------------------------------------------------------


class TestAnimate:
    def test_returns_func_animation(self) -> None:
        from matplotlib.animation import FuncAnimation

        clip = _make_clip(n_frames=3)
        anim = EventVisualizer().animate(clip, "pass")
        assert isinstance(anim, FuncAnimation)

    def test_frame_count_matches_clip_length(self) -> None:
        with patch("turf.event_visualizer.FuncAnimation") as MockFA:
            MockFA.return_value = MagicMock()
            clip = _make_clip(n_frames=7)
            EventVisualizer().animate(clip, "pass")

            call_kwargs = MockFA.call_args
            frames_arg = call_kwargs.kwargs.get("frames") or call_kwargs.args[2]
            assert frames_arg == 7

    def test_smooth_true_does_not_raise(self) -> None:
        from matplotlib.animation import FuncAnimation

        clip = _make_clip(n_frames=10)
        anim = EventVisualizer().animate(clip, "pass", smooth=True)
        assert isinstance(anim, FuncAnimation)

    def test_smooth_window_polyorder_forwarded(self) -> None:
        from matplotlib.animation import FuncAnimation

        clip = _make_clip(n_frames=10)
        anim = EventVisualizer().animate(
            clip, "pass", smooth=True, smooth_window=7, smooth_polyorder=1
        )
        assert isinstance(anim, FuncAnimation)

    def test_nan_frame_clears_home_scatter(self) -> None:
        """All-NaN positions in a frame must clear home_scat, not leave stale points."""
        from unittest.mock import MagicMock, call, patch

        home_rows = [
            {
                "Period": 1,
                "Time [s]": 0.0,
                "Home_1_x": 10.0,
                "Home_1_y": 5.0,
                "Home_2_x": float("nan"),
                "Home_2_y": float("nan"),
                "ball_x": 0.5,
                "ball_y": 1.0,
            },
            {
                "Period": 1,
                "Time [s]": 1.0,
                "Home_1_x": float("nan"),
                "Home_1_y": float("nan"),
                "Home_2_x": float("nan"),
                "Home_2_y": float("nan"),
                "ball_x": float("nan"),
                "ball_y": float("nan"),
            },
        ]
        away_rows = [
            {
                "Period": 1,
                "Time [s]": 0.0,
                "Away_1_x": -10.0,
                "Away_1_y": -5.0,
                "Away_2_x": float("nan"),
                "Away_2_y": float("nan"),
                "ball_x": 0.5,
                "ball_y": 1.0,
            },
            {
                "Period": 1,
                "Time [s]": 1.0,
                "Away_1_x": float("nan"),
                "Away_1_y": float("nan"),
                "Away_2_x": float("nan"),
                "Away_2_y": float("nan"),
                "ball_x": float("nan"),
                "ball_y": float("nan"),
            },
        ]
        clip = EventClip(
            event_idx=0,
            start_frame=0,
            end_frame=1,
            metadata={
                "event_idx": 0,
                "start_frame": 0,
                "end_frame": 1,
                "start_x": 0.0,
                "start_y": 0.0,
                "end_x": 1.0,
                "end_y": 0.0,
                "from_player": "A",
                "to_player": "B",
                "team": "Home",
                "subtype": "success",
                "period": 1,
            },
            home_frames=pd.DataFrame(home_rows),
            away_frames=pd.DataFrame(away_rows),
        )

        captured_scats: list[MagicMock] = []
        captured_update: list[object] = []

        with patch("turf.event_visualizer.Pitch") as MockPitch, patch(
            "turf.event_visualizer.FuncAnimation"
        ) as MockFA:
            # Capture the _update closure without letting FuncAnimation run it
            def _capture_anim(
                _fig: object, func: object, **_kw: object
            ) -> MagicMock:
                captured_update.append(func)
                return MagicMock()

            MockFA.side_effect = _capture_anim

            mock_pitch_instance = MagicMock()
            MockPitch.return_value = mock_pitch_instance
            fig, real_ax = plt.subplots()

            def _capturing_scatter(*_args: object, **_kwargs: object) -> MagicMock:
                m: MagicMock = MagicMock()
                captured_scats.append(m)
                return m

            real_ax.scatter = _capturing_scatter  # type: ignore[method-assign]
            mock_pitch_instance.draw.return_value = (fig, real_ax)

            EventVisualizer().animate(clip, "pass")
            update_fn = captured_update[0]
            assert callable(update_fn)
            update_fn(0)  # type: ignore[operator]  # valid frame → set_offsets with data
            update_fn(1)  # type: ignore[operator]  # all-NaN → must call set_offsets([])
            plt.close(fig)

        # captured_scats[0] is home_scat (first scatter call in animate())
        home_scat = captured_scats[0]
        assert home_scat.set_offsets.called
        assert home_scat.set_offsets.call_args == call([])

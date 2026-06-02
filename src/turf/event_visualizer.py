from __future__ import annotations

import math
from typing import TYPE_CHECKING

from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]
    from matplotlib.figure import Figure

    from turf.event_extractor import EventClip

_PITCH_TYPE = "skillcorner"
_PITCH_LENGTH = 105  # metres; skillcorner pitch is centred at origin, ±52.5 × ±34
_PITCH_WIDTH = 68

_SMOOTH_WINDOW = 5  # frames; at 25 fps, this is 0.2 seconds; try (5, 7, 9)
_SMOOTH_POLYORDER = 2


class EventVisualizer:
    def freeze_frame(
        self,
        clip: EventClip,
        label: str,
        smooth: bool = False,
        smooth_window: int = _SMOOTH_WINDOW,
        smooth_polyorder: int = _SMOOTH_POLYORDER,
    ) -> Figure:
        """Return a freeze-frame figure of the event start position."""
        pitch = Pitch(
            pitch_type=_PITCH_TYPE,
            pitch_length=_PITCH_LENGTH,
            pitch_width=_PITCH_WIDTH,
        )
        fig, ax = pitch.draw()

        home_frames = (
            self._smooth_frames(clip.home_frames, smooth_window, smooth_polyorder)
            if smooth
            else clip.home_frames
        )
        away_frames = (
            self._smooth_frames(clip.away_frames, smooth_window, smooth_polyorder)
            if smooth
            else clip.away_frames
        )

        row_home = home_frames.iloc[0]
        row_away = away_frames.iloc[0]

        hx, hy = self._player_xy(row_home, "Home")
        ax_x, ay = self._player_xy(row_away, "Away")
        bx, by = self._ball_xy(row_home)

        if hx:
            pitch.scatter(hx, hy, ax=ax, color="royalblue", s=120, zorder=3)
        if ax_x:
            pitch.scatter(ax_x, ay, ax=ax, color="tomato", s=120, zorder=3)
        if bx is not None:
            pitch.scatter(
                [bx], [by], ax=ax, color="white", edgecolors="black", s=80, zorder=4
            )

        if label == "pass":
            sx = float(clip.metadata["start_x"])  # type: ignore[arg-type]
            sy = float(clip.metadata["start_y"])  # type: ignore[arg-type]
            ex = float(clip.metadata["end_x"])  # type: ignore[arg-type]
            ey = float(clip.metadata["end_y"])  # type: ignore[arg-type]
            pitch.arrows(
                sx, sy, ex, ey, ax=ax, color="yellow", width=2, headwidth=5, zorder=5
            )

        ts = self._timestamp_str(clip)
        ax.set_title(f"{label.capitalize()} | {ts}", fontsize=12, pad=10)

        fig.tight_layout()
        return fig  # type: ignore[no-any-return]

    def animate(
        self,
        clip: EventClip,
        label: str,
        fps: float = 25.0,
        smooth: bool = False,
        smooth_window: int = _SMOOTH_WINDOW,
        smooth_polyorder: int = _SMOOTH_POLYORDER,
    ) -> FuncAnimation:
        """Return a FuncAnimation stepping through every tracking frame in the clip."""
        pitch = Pitch(
            pitch_type=_PITCH_TYPE,
            pitch_length=_PITCH_LENGTH,
            pitch_width=_PITCH_WIDTH,
        )
        fig, ax = pitch.draw()

        home_frames = (
            self._smooth_frames(clip.home_frames, smooth_window, smooth_polyorder)
            if smooth
            else clip.home_frames
        )
        away_frames = (
            self._smooth_frames(clip.away_frames, smooth_window, smooth_polyorder)
            if smooth
            else clip.away_frames
        )

        n_frames = len(home_frames)
        home_scat = ax.scatter([], [], color="royalblue", s=120, zorder=3)
        away_scat = ax.scatter([], [], color="tomato", s=120, zorder=3)
        ball_scat = ax.scatter(
            [], [], color="white", edgecolors="black", s=80, zorder=4
        )
        title = ax.set_title("")

        def _update(frame_i: int) -> tuple[object, ...]:
            row_home = home_frames.iloc[frame_i]
            row_away = away_frames.iloc[frame_i]

            hx, hy = self._player_xy(row_home, "Home")
            ax_x, ay = self._player_xy(row_away, "Away")
            bx, by = self._ball_xy(row_home)

            if hx:
                home_scat.set_offsets(list(zip(hx, hy, strict=True)))
            if ax_x:
                away_scat.set_offsets(list(zip(ax_x, ay, strict=True)))
            if bx is not None:
                ball_scat.set_offsets([[bx, by]])

            ts = self._frame_timestamp_str(row_home, clip.metadata["period"])
            title.set_text(f"{label.capitalize()} | {ts}")
            return home_scat, away_scat, ball_scat, title

        return FuncAnimation(
            fig,
            _update,  # type: ignore[arg-type]
            frames=n_frames,
            interval=1000.0 / fps,
            blit=True,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _smooth_frames(
        self,
        frames: pd.DataFrame,
        window: int = _SMOOTH_WINDOW,
        polyorder: int = _SMOOTH_POLYORDER,
    ) -> pd.DataFrame:
        """Apply Savitzky-Golay smoothing to all position columns."""
        from scipy.signal import savgol_filter  # type: ignore[import-untyped]

        result = frames.copy()
        coord_cols = [c for c in frames.columns if c.endswith(("_x", "_y"))]
        # window_length must be odd and ≤ number of valid data points
        w = window if window % 2 == 1 else window - 1
        for col in coord_cols:
            series = result[col]
            valid = series.notna()
            if valid.sum() < w:
                continue
            result.loc[valid, col] = savgol_filter(
                series[valid].to_numpy(),
                window_length=w,
                polyorder=polyorder,
            )
        return result

    def _player_xy(
        self, row: pd.Series, prefix: str
    ) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        n = 1
        while True:
            xc, yc = f"{prefix}_{n}_x", f"{prefix}_{n}_y"
            if xc not in row.index:
                break
            try:
                xf, yf = float(row[xc]), float(row[yc])
            except (TypeError, ValueError):
                n += 1
                continue
            if not (math.isnan(xf) or math.isnan(yf)):
                xs.append(xf)
                ys.append(yf)
            n += 1
        return xs, ys

    def _ball_xy(self, row: pd.Series) -> tuple[float | None, float | None]:
        try:
            bx, by = float(row["ball_x"]), float(row["ball_y"])
        except (KeyError, TypeError, ValueError):
            return None, None
        if math.isnan(bx) or math.isnan(by):
            return None, None
        return bx, by

    def _timestamp_str(self, clip: EventClip) -> str:
        return self._frame_timestamp_str(
            clip.home_frames.iloc[0], clip.metadata["period"]
        )

    def _frame_timestamp_str(self, row: pd.Series, period: object) -> str:
        seconds = int(float(row["Time [s]"]))
        m, s = divmod(seconds, 60)
        half = "H1" if int(str(period)) == 1 else "H2"
        return f"{half} {m:02d}:{s:02d}"

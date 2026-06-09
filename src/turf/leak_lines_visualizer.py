from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.text import Text
from mplsoccer import Pitch  # type: ignore[import-untyped]

from leak.lines import smooth_line_assignments

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

_PITCH_TYPE = "skillcorner"
_PITCH_LENGTH = 105
_PITCH_WIDTH = 68
_HALF_WIDTH = _PITCH_WIDTH / 2  # 34.0
_YLIM_BOTTOM = -42.0
_LABEL_Y = -38.0

_LINE_COLORS: dict[int, str] = {
    1: "#4e79a7",
    2: "#59a14f",
    3: "#f28e2b",
    4: "#e15759",
}
_TEAM_COLORS: dict[str, str] = {
    "Home": "royalblue",
    "Away": "tomato",
}
_GK_EDGE_COLOR = "lime"
_MAX_LINES = 4
_TRAIL_ALPHA_MIN = 0.08
_TRAIL_ALPHA_MAX = 0.65
_BALL_COLOR = "#FFA500"  # orange — visible on both dark and light pitches
_TRAIL_RGB = (1.0, 0.647, 0.0)  # orange matching ball
_GAP_TEXT_Y = 28.0  # y position of debug gap labels (near top of pitch)


class LeakLinesVisualizer:
    def animate(
        self,
        defending_frames: pd.DataFrame,
        attacking_frames: pd.DataFrame,
        defending_team: str,
        attacking_team: str,
        metadata: dict[str, object],
        fps: float = 25.0,
        smooth_lines: bool = True,
        debug: bool = False,
    ) -> FuncAnimation:
        """Return a FuncAnimation stepping through every tracking frame."""
        if smooth_lines:
            defending_frames = smooth_line_assignments(defending_frames, defending_team)

        # Defending GK x — determines which direction is "deep"
        first_def_row = defending_frames.iloc[0]
        gk_id = self._find_player_gk(first_def_row, defending_team)
        def_gk_x: float = 0.0
        if gk_id is not None:
            gk_xcol = f"{defending_team}_{gk_id}_x"
            with contextlib.suppress(KeyError, TypeError, ValueError):
                def_gk_x = float(first_def_row[gk_xcol])

        pitch = Pitch(
            pitch_type=_PITCH_TYPE,
            pitch_length=_PITCH_LENGTH,
            pitch_width=_PITCH_WIDTH,
        )
        fig, ax = pitch.draw()
        ax.set_ylim(_YLIM_BOTTOM, _HALF_WIDTH)

        n_frames = min(len(defending_frames), len(attacking_frames))

        # Attacking GK ID — stable across frames, derived from first frame
        atk_gk_id = self._find_player_gk(attacking_frames.iloc[0], attacking_team)

        # Ball trail: pre-extract all positions (None where ball absent)
        ball_trail: list[tuple[float, float] | None] = [
            self._ball_xy_pair(defending_frames.iloc[i]) for i in range(n_frames)
        ]

        # Starting ball position and which unit line it is already behind
        start_ball: tuple[float, float] | None = ball_trail[0] if ball_trail else None
        start_label = ""
        if start_ball is not None:
            lp0 = self._players_by_line(defending_frames.iloc[0], defending_team)
            repr_xs_0 = {
                ln: self._repr_x(lp0[ln], def_gk_x)
                for ln in range(1, _MAX_LINES + 1)
                if lp0.get(ln)
            }
            start_label = self._ball_line_label(start_ball[0], repr_xs_0)

        # ── artists ─────────────────────────────────────────────────────────
        atk_color = _TEAM_COLORS.get(attacking_team, "gray")
        atk_outfield_scat = ax.scatter(
            [], [], color=atk_color, s=120, alpha=0.5, zorder=3
        )
        atk_gk_scat = ax.scatter(
            [],
            [],
            color=atk_color,
            marker="D",
            s=160,
            edgecolors=_GK_EDGE_COLOR,
            linewidths=2,
            alpha=0.5,
            zorder=5,
        )

        def_color = _TEAM_COLORS.get(defending_team, "gray")
        def_outfield_scat = ax.scatter([], [], color=def_color, s=140, zorder=4)
        def_gk_scat = ax.scatter(
            [],
            [],
            color=def_color,
            marker="D",
            s=160,
            edgecolors=_GK_EDGE_COLOR,
            linewidths=2,
            zorder=5,
        )

        # Live ball marker
        ball_scat = ax.scatter(
            [], [], color=_BALL_COLOR, edgecolors="black", s=80, zorder=6
        )

        # Ball trail (grows each frame, oldest = most transparent)
        trail_scat = ax.scatter([], [], s=18, edgecolors="none", zorder=5)

        # ── debug: inter-line gap labels ─────────────────────────────────────
        # Pre-allocated text artists between adjacent repr lines.
        # Visible only when debug=True and both neighbouring lines exist.
        gap_texts: dict[int, Text] = {}
        if debug:
            for ln in range(1, _MAX_LINES):
                gap_texts[ln] = ax.text(
                    0.0,
                    _GAP_TEXT_Y,
                    "",
                    color="black",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.7, "boxstyle": "round"},
                    visible=False,
                    zorder=10,
                )

        # ── static: freeze-frame starting ball position ──────────────────────
        # These are drawn before FuncAnimation so they become part of the
        # background (blit=True) and are always visible without re-drawing.
        if start_ball is not None:
            sbx, sby = start_ball
            ax.scatter(
                [sbx],
                [sby],
                color="yellow",
                edgecolors="darkorange",
                marker="*",
                s=140,
                linewidths=1.5,
                zorder=7,
            )
            if start_label:
                ax.text(
                    sbx + 1.2,
                    sby + 1.5,
                    start_label,
                    color="darkorange",
                    fontsize=9,
                    fontweight="bold",
                    zorder=8,
                    ha="left",
                    va="bottom",
                )

        # ── per-line: zig-zag, repr dashed line, text label ─────────────────
        zigzag_lines = {}
        repr_lines = {}
        line_labels = {}
        for ln in range(1, _MAX_LINES + 1):
            (zz,) = ax.plot(
                [], [], color=_LINE_COLORS[ln], linewidth=1.5, alpha=0.7, zorder=3
            )
            zigzag_lines[ln] = zz
            (rl,) = ax.plot(
                [],
                [],
                color=_LINE_COLORS[ln],
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
                zorder=2,
            )
            repr_lines[ln] = rl
            line_labels[ln] = ax.text(
                0.0,
                _LABEL_Y,
                "",
                color=_LINE_COLORS[ln],
                fontsize=8,
                rotation=45,
                ha="center",
                va="top",
                visible=False,
            )

        # ── legend ───────────────────────────────────────────────────────────
        legend_handles = [
            Line2D(
                [],
                [],
                color=def_color,
                marker="o",
                markersize=7,
                linewidth=0,
                label=f"{defending_team} (def.)",
            ),
            Line2D(
                [],
                [],
                color=def_color,
                marker="D",
                markersize=7,
                linewidth=0,
                markeredgecolor=_GK_EDGE_COLOR,
                markeredgewidth=1.5,
                label=f"{defending_team} GK",
            ),
            Line2D(
                [],
                [],
                color=atk_color,
                marker="o",
                markersize=7,
                linewidth=0,
                alpha=0.5,
                label=f"{attacking_team} (atk.)",
            ),
            Line2D(
                [],
                [],
                color=atk_color,
                marker="D",
                markersize=7,
                linewidth=0,
                markeredgecolor=_GK_EDGE_COLOR,
                markeredgewidth=1.5,
                alpha=0.5,
                label=f"{attacking_team} GK",
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

        title = ax.set_title("")
        period = metadata.get("period", 1)

        def _update(frame_i: int) -> tuple[object, ...]:
            row_def = defending_frames.iloc[frame_i]
            row_atk = attacking_frames.iloc[frame_i]

            # Attacking team — split GK from outfield
            atk_all = self._player_xy_with_ids(row_atk, attacking_team)
            atk_out = [(x, y) for pid, x, y in atk_all if pid != atk_gk_id]
            atk_gk = [(x, y) for pid, x, y in atk_all if pid == atk_gk_id]

            atk_outfield_scat.set_offsets(atk_out if atk_out else np.empty((0, 2)))
            atk_gk_scat.set_offsets(atk_gk if atk_gk else np.empty((0, 2)))

            # Live ball
            bx, by = self._ball_xy(row_def)
            if bx is not None:
                ball_scat.set_offsets([[bx, by]])
            else:
                ball_scat.set_offsets(np.empty((0, 2)))

            # Ball trail with alpha gradient (oldest = most transparent)
            valid_trail = [p for p in ball_trail[: frame_i + 1] if p is not None]
            if valid_trail:
                n_t = len(valid_trail)
                alphas = (
                    np.array([_TRAIL_ALPHA_MAX])
                    if n_t == 1
                    else np.linspace(_TRAIL_ALPHA_MIN, _TRAIL_ALPHA_MAX, n_t)
                )
                trail_rgba = np.empty((n_t, 4))
                trail_rgba[:, 0] = _TRAIL_RGB[0]
                trail_rgba[:, 1] = _TRAIL_RGB[1]
                trail_rgba[:, 2] = _TRAIL_RGB[2]
                trail_rgba[:, 3] = alphas
                trail_scat.set_offsets(valid_trail)
                trail_scat.set_facecolor(trail_rgba)
            else:
                trail_scat.set_offsets(np.empty((0, 2)))

            # Defending team
            line_players = self._players_by_line(row_def, defending_team)

            gk_players = line_players.get(0, [])
            def_gk_scat.set_offsets(gk_players if gk_players else np.empty((0, 2)))

            all_outfield = [
                p for ln in range(1, _MAX_LINES + 1) for p in line_players.get(ln, [])
            ]
            def_outfield_scat.set_offsets(
                all_outfield if all_outfield else np.empty((0, 2))
            )

            # Repr x per active line; sort deepest first (highest abs) for L1=deepest
            repr_xs_active: dict[int, float] = {
                ln: self._repr_x(line_players[ln], def_gk_x)
                for ln in range(1, _MAX_LINES + 1)
                if line_players.get(ln)
            }
            sorted_by_depth = sorted(
                repr_xs_active, key=lambda ln: abs(repr_xs_active[ln]), reverse=True
            )
            ln_to_user: dict[int, int] = {
                ln: i + 1 for i, ln in enumerate(sorted_by_depth)
            }

            for ln in range(1, _MAX_LINES + 1):
                players = line_players.get(ln, [])
                if players:
                    if len(players) >= 2:
                        sorted_p = sorted(players, key=lambda p: p[1])
                        zigzag_lines[ln].set_data(
                            [p[0] for p in sorted_p], [p[1] for p in sorted_p]
                        )
                    else:
                        zigzag_lines[ln].set_data([], [])
                    user_ln = ln_to_user[ln]
                    color = _LINE_COLORS[user_ln]
                    rx = repr_xs_active[ln]
                    zigzag_lines[ln].set_color(color)
                    repr_lines[ln].set_data([rx, rx], [-_HALF_WIDTH, _HALF_WIDTH])
                    repr_lines[ln].set_color(color)
                    line_labels[ln].set_position((rx, _LABEL_Y))
                    line_labels[ln].set_text(f"L{user_ln}")
                    line_labels[ln].set_color(color)
                    line_labels[ln].set_visible(True)
                else:
                    zigzag_lines[ln].set_data([], [])
                    repr_lines[ln].set_data([], [])
                    line_labels[ln].set_visible(False)

            # Debug: inter-line gap in metres between adjacent repr lines
            if debug:
                for ln in range(1, _MAX_LINES):
                    if ln in repr_xs_active and ln + 1 in repr_xs_active:
                        rx1 = repr_xs_active[ln]
                        rx2 = repr_xs_active[ln + 1]
                        gap = abs(rx2 - rx1)
                        mid_x = (rx1 + rx2) / 2.0
                        gap_texts[ln].set_position((mid_x, _GAP_TEXT_Y))
                        gap_texts[ln].set_text(f"{gap:.1f}m")
                        gap_texts[ln].set_visible(True)
                    else:
                        gap_texts[ln].set_visible(False)
            ts = self._frame_timestamp_str(row_def, period)
            title.set_text(f"Unit Lines | {ts}")

            return (
                atk_outfield_scat,
                atk_gk_scat,
                trail_scat,
                ball_scat,
                def_outfield_scat,
                def_gk_scat,
                *zigzag_lines.values(),
                *repr_lines.values(),
                *line_labels.values(),
                *gap_texts.values(),
                title,
            )

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

    def _find_player_gk(self, row: pd.Series, team: str) -> str | None:
        """Return the player ID with the highest abs(x) — the goalkeeper."""
        best_id: str | None = None
        best_abs = -1.0
        n = 1
        while True:
            xc = f"{team}_{n}_x"
            if xc not in row.index:
                break
            try:
                xval = float(row[xc])
            except (TypeError, ValueError):
                n += 1
                continue
            a = abs(xval)
            if not math.isnan(a) and a > best_abs:
                best_abs = a
                best_id = str(n)
            n += 1
        return best_id

    @staticmethod
    def _ball_line_label(ball_x: float, line_repr_xs: dict[int, float]) -> str:
        """Return 'L{n}' for the deepest line beaten (L1=most dangerous/deepest).

        Ranks lines by abs(repr_x) descending: highest abs = closest to GK = L1.
        """
        beaten_lns = {
            ln: abs(rx)
            for ln, rx in line_repr_xs.items()
            if ball_x * rx > 0 and abs(ball_x) >= abs(rx)
        }
        if not beaten_lns:
            return ""
        deepest_ln = max(beaten_lns, key=lambda ln: beaten_lns[ln])
        all_sorted = sorted(
            line_repr_xs, key=lambda ln: abs(line_repr_xs[ln]), reverse=True
        )
        user_rank = all_sorted.index(deepest_ln) + 1
        return f"L{user_rank}"

    @staticmethod
    def _repr_x(players: list[tuple[float, float]], gk_x: float) -> float:
        """Return x of the player closest to the defending goal (deepest in line)."""
        sign = math.copysign(1.0, gk_x)
        return max((p[0] for p in players), key=lambda x: x * sign)

    def _player_xy_with_ids(
        self, row: pd.Series, team: str
    ) -> list[tuple[str, float, float]]:
        players: list[tuple[str, float, float]] = []
        n = 1
        while True:
            xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
            if xc not in row.index:
                break
            try:
                xf, yf = float(row[xc]), float(row[yc])
            except (TypeError, ValueError):
                n += 1
                continue
            if not (math.isnan(xf) or math.isnan(yf)):
                players.append((str(n), xf, yf))
            n += 1
        return players

    def _player_ids_by_line(self, row: pd.Series, team: str) -> dict[int, list[str]]:
        result: dict[int, list[str]] = {}
        n = 1
        while True:
            xc = f"{team}_{n}_x"
            if xc not in row.index:
                break
            line_col = f"{team}_{n}_line"
            try:
                x = float(row[xc])
                ln_raw = float(row[line_col]) if line_col in row.index else float("nan")
            except (TypeError, ValueError):
                n += 1
                continue
            if not (math.isnan(x) or math.isnan(ln_raw)):
                result.setdefault(int(ln_raw), []).append(str(n))
            n += 1
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

    def _ball_xy_pair(self, row: pd.Series) -> tuple[float, float] | None:
        bx, by = self._ball_xy(row)
        if bx is not None and by is not None:
            return (bx, by)
        return None

    def _ball_xy(self, row: pd.Series) -> tuple[float | None, float | None]:
        try:
            bx, by = float(row["ball_x"]), float(row["ball_y"])
        except (KeyError, TypeError, ValueError):
            return None, None
        if math.isnan(bx) or math.isnan(by):
            return None, None
        return bx, by

    def _players_by_line(
        self, row: pd.Series, team: str
    ) -> dict[int, list[tuple[float, float]]]:
        result: dict[int, list[tuple[float, float]]] = {}
        n = 1
        while True:
            xc = f"{team}_{n}_x"
            if xc not in row.index:
                break
            yc = f"{team}_{n}_y"
            line_col = f"{team}_{n}_line"
            try:
                x = float(row[xc])
                y = float(row[yc])
                ln_raw = float(row[line_col]) if line_col in row.index else float("nan")
            except (TypeError, ValueError):
                n += 1
                continue
            if not (math.isnan(x) or math.isnan(y) or math.isnan(ln_raw)):
                line_num = int(ln_raw)
                result.setdefault(line_num, []).append((x, y))
            n += 1
        return result

    def _frame_timestamp_str(self, row: pd.Series, period: object) -> str:
        seconds = int(float(row["Time [s]"]))
        m, s = divmod(seconds, 60)
        half = "H1" if int(str(period)) == 1 else "H2"
        return f"{half} {m:02d}:{s:02d}"

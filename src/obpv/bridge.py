"""Convert wide-format tracking DataFrames into GameState objects."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

from obpv._state import GameState, PlayerState

__all__ = ["frames_to_game_state"]

_MAX_PLAYERS: int = 16


def _player_states(
    df: pd.DataFrame,
    frame_idx: int,
    prefix: str,
    team_id: str,
) -> list[PlayerState]:
    """Parse one team's players from a wide-format tracking DataFrame."""
    n = len(df)
    row = df.iloc[frame_idx]

    # Determine adjacent row and Δt for velocity estimation.
    if n == 1:
        adj_row: pd.Series | None = None
        dt = 1.0
        forward = True
    elif frame_idx < n - 1:
        adj_row = df.iloc[frame_idx + 1]
        dt = float(adj_row["Time [s]"]) - float(row["Time [s]"])
        forward = True
    else:
        adj_row = df.iloc[frame_idx - 1]
        dt = float(row["Time [s]"]) - float(adj_row["Time [s]"])
        forward = False

    players: list[PlayerState] = []
    for i in range(1, _MAX_PLAYERS + 1):
        x_col = f"{prefix}_{i}_x"
        y_col = f"{prefix}_{i}_y"
        if x_col not in df.columns or y_col not in df.columns:
            continue
        px, py = row.get(x_col), row.get(y_col)
        if pd.isna(px) or pd.isna(py):
            continue

        pos: npt.NDArray[np.float64] = np.array([float(px), float(py)], dtype=float)

        vel: npt.NDArray[np.float64] = np.zeros(2, dtype=float)
        if adj_row is not None and dt > 0:
            ax, ay = adj_row.get(x_col), adj_row.get(y_col)
            if not pd.isna(ax) and not pd.isna(ay):
                delta = np.array(
                    [float(ax) - float(px), float(ay) - float(py)], dtype=float
                )
                vel = delta / dt if forward else -delta / dt

        players.append(
            PlayerState(
                player_id=f"{prefix}_{i}",
                team_id=team_id,
                position=pos,
                velocity=vel,
            )
        )

    return players


def frames_to_game_state(
    home_df: pd.DataFrame,
    away_df: pd.DataFrame,
    frame_idx: int,
    attacking_team_id: str,
    attack_dir: int,
) -> GameState:
    """Build a GameState from wide-format per-pass tracking DataFrames.

    Parameters
    ----------
    home_df, away_df:
        Wide-format DataFrames with columns ``Home_N_x``, ``Home_N_y`` /
        ``Away_N_x``, ``Away_N_y``, ``ball_x``, ``ball_y``, ``Time [s]``.
    frame_idx:
        Row index to use (0 = pass start, ``len-1`` = pass end).
    attacking_team_id:
        ``"Home"`` or ``"Away"`` — which team made the pass.
    attack_dir:
        ``+1`` if the attacking team plays left-to-right in SkillCorner coords,
        ``-1`` if right-to-left.
    """
    home_players = _player_states(home_df, frame_idx, "Home", "Home")
    away_players = _player_states(away_df, frame_idx, "Away", "Away")

    row = home_df.iloc[frame_idx]
    ball_pos: npt.NDArray[np.float64] = np.array(
        [float(row["ball_x"]), float(row["ball_y"])], dtype=float
    )

    defending_team_id = "Away" if attacking_team_id == "Home" else "Home"
    attacking_direction = "left_to_right" if attack_dir > 0 else "right_to_left"

    return GameState(
        ball_position=ball_pos,
        players=home_players + away_players,
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        metadata={"attacking_direction": attacking_direction},
    )

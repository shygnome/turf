from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class PlayerState:
    player_id: str
    team_id: str
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    is_goalkeeper: bool = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        if self.position.shape != (2,):
            raise ValueError("position must have shape (2,)")
        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.velocity.shape != (2,):
            raise ValueError("velocity must have shape (2,)")

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    @property
    def vx(self) -> float:
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        return float(self.velocity[1])


@dataclass
class GameState:
    ball_position: npt.NDArray[np.float64]
    players: list[PlayerState]
    frame_id: int | None = None
    possessing_team_id: str | None = None
    attacking_team_id: str | None = None
    defending_team_id: str | None = None
    period: int | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ball_position = np.asarray(self.ball_position, dtype=float)
        if self.ball_position.shape != (2,):
            raise ValueError("ball_position must have shape (2,)")
        if len(self.players) == 0:
            raise ValueError("players must not be empty")

    @property
    def ball_x(self) -> float:
        return float(self.ball_position[0])

    @property
    def ball_y(self) -> float:
        return float(self.ball_position[1])

    @property
    def attacking_players(self) -> list[PlayerState]:
        if self.attacking_team_id is None:
            return []
        return [p for p in self.players if p.team_id == self.attacking_team_id]

    @property
    def defending_players(self) -> list[PlayerState]:
        if self.defending_team_id is None:
            return []
        return [p for p in self.players if p.team_id == self.defending_team_id]

    def players_for_team(self, team_id: str) -> list[PlayerState]:
        return [p for p in self.players if p.team_id == team_id]


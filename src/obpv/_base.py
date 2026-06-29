from __future__ import annotations

from abc import ABC, abstractmethod

from obpv._grid import PitchGrid
from obpv._state import GameState
from obpv._surface import PitchSurface


class PitchSurfaceModel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        raise NotImplementedError

    def get_config(self) -> dict[str, object]:
        return {}


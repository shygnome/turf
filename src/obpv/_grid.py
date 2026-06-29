from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PitchGrid:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    length: float = 105.0
    width: float = 68.0
    origin: str = "center"

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y), len(self.x))

    @property
    def meshgrid(self) -> list[npt.NDArray[np.float64]]:
        xx, yy = np.meshgrid(self.x, self.y)
        return [xx, yy]

    @property
    def dx(self) -> float:
        return self.length / (len(self.x) - 1)

    @property
    def dy(self) -> float:
        return self.width / (len(self.y) - 1)

    @property
    def home_goal(self) -> tuple[float, float]:
        return (-self.length / 2, 0)

    @property
    def away_goal(self) -> tuple[float, float]:
        return (self.length / 2, 0)

    @classmethod
    def from_resolution(
        cls,
        length: float = 105.0,
        width: float = 68.0,
        n_x: int = 50,
        n_y: int = 32,
    ) -> PitchGrid:
        x = np.linspace(-length / 2, length / 2, n_x)
        y = np.linspace(-width / 2, width / 2, n_y)
        return cls(x=x, y=y, length=length, width=width, origin="center")


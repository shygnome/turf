from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from obpv._grid import PitchGrid


@dataclass
class PitchSurface:
    values: npt.NDArray[np.float64]
    grid: PitchGrid
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != self.grid.shape:
            raise ValueError(
                f"values shape {self.values.shape} does not match "
                f"grid shape {self.grid.shape}"
            )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def x(self) -> npt.NDArray[np.float64]:
        return self.grid.x

    @property
    def y(self) -> npt.NDArray[np.float64]:
        return self.grid.y

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values.tolist(),
            "grid": {
                "x": self.grid.x.tolist(),
                "y": self.grid.y.tolist(),
                "length": self.grid.length,
                "width": self.grid.width,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PitchSurface:
        g = data["grid"]
        grid = PitchGrid(
            x=np.array(g["x"]),
            y=np.array(g["y"]),
            length=g["length"],
            width=g["width"],
        )
        return cls(
            values=np.array(data["values"]),
            grid=grid,
            name=data["name"],
            metadata=data.get("metadata", {}),
        )


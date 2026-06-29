"""Transition models: how likely is the ball to move to each pitch location."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

from obpv._base import PitchSurfaceModel
from obpv._grid import PitchGrid
from obpv._state import GameState
from obpv._surface import PitchSurface


class DummyTransitionModel(PitchSurfaceModel):
    def __init__(self, constant_value: float = 1.0) -> None:
        self.constant_value = constant_value

    @property
    def name(self) -> str:
        return "dummy_transition"

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        return PitchSurface(
            values=np.full(grid.shape, self.constant_value, dtype=float),
            grid=grid,
            name=self.name,
        )

    def get_config(self) -> dict[str, object]:
        return {"constant_value": self.constant_value}


class TransitionGaussModel(PitchSurfaceModel):
    """Gaussian kernel centred on the ball -- no data files required."""

    def __init__(self, sigma: float = 15.0) -> None:
        self.sigma = sigma

    @property
    def name(self) -> str:
        return "transition_gauss"

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        ball_x, ball_y = state.ball_position
        xx, yy = grid.meshgrid
        distances_sq = (xx - ball_x) ** 2 + (yy - ball_y) ** 2
        values = np.exp(-distances_sq / (2 * self.sigma**2))
        return PitchSurface(
            values=values,
            grid=grid,
            name=self.name,
            metadata={
                "ball_position": state.ball_position.tolist(),
                "sigma": self.sigma,
            },
        )

    def get_config(self) -> dict[str, object]:
        return {"sigma": self.sigma}


def _crop_transition_to_grid(
    matrix: npt.NDArray[np.float64], ball_x: float, ball_y: float
) -> npt.NDArray[np.float64]:
    """Extract a (32, 50) window from a (64, 100) transition matrix.

    The matrix is twice the pitch resolution. A sliding window centred on the
    ball's grid position is extracted so the output represents the pass-direction
    distribution relative to the ball's current location.
    """
    ball_grid_x = int(np.clip((ball_x + 52.5) // (105 / 50), 0, 49))
    ball_grid_y = int(np.clip((ball_y + 34) // (68 / 32), 0, 31))
    return matrix[
        31 - ball_grid_y: 63 - ball_grid_y,
        49 - ball_grid_x: 99 - ball_grid_x,
    ]


class CsvTransitionModel(PitchSurfaceModel):
    """Transition model from a pre-computed 64x100 CSV grid (Transition_gauss.csv).

    A (32, 50) window centred on the ball is extracted per frame.
    """

    def __init__(
        self, matrix: npt.NDArray[np.float64], normalize: bool = False
    ) -> None:
        self._matrix = matrix.astype(float)
        if normalize:
            m = self._matrix.max()
            if m > 0:
                self._matrix /= m

    @classmethod
    def from_csv(
        cls, csv_path: str | Path, normalize: bool = True
    ) -> CsvTransitionModel:
        df = pd.read_csv(csv_path, header=None)
        return cls(matrix=np.array(df, dtype=float), normalize=normalize)

    @property
    def name(self) -> str:
        return "csv_transition"

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        ball_x, ball_y = float(state.ball_position[0]), float(state.ball_position[1])
        values = _crop_transition_to_grid(self._matrix, ball_x, ball_y)

        if values.shape != grid.shape:
            from scipy.ndimage import zoom  # type: ignore[import-untyped]
            zoom_factors = (
                grid.shape[0] / values.shape[0],
                grid.shape[1] / values.shape[1],
            )
            values = zoom(values, zoom_factors, order=1)

        return PitchSurface(
            values=values,
            grid=grid,
            name=self.name,
            metadata={"ball_position": state.ball_position.tolist()},
        )


class _ZoneTransitionKernel:
    """18-zone KDE transition kernel -- load and predict only (no fit)."""

    def __init__(self) -> None:
        self.transition_distributions: dict[int, pd.DataFrame] = {}

    def load_from_csv(self, load_dir: str | Path) -> None:
        load_dir = Path(load_dir)
        for file_name in os.listdir(load_dir):
            if file_name.startswith("Area") and file_name.endswith("_Transition.csv"):
                area = int(file_name[len("Area"): -len("_Transition.csv")])
                df = pd.read_csv(load_dir / file_name, header=None)
                self.transition_distributions[area] = df

    def predict(self, ball_x: float, ball_y: float) -> pd.DataFrame | None:
        if not self.transition_distributions:
            raise ValueError(
                "No transition distributions loaded. Call load_from_csv first."
            )
        area = self._divide_pitch(ball_x, ball_y)
        return self.transition_distributions.get(area)

    @staticmethod
    def _divide_pitch(
        x: float, y: float, dimensions: tuple[float, float] = (105.0, 68.0)
    ) -> int:
        length, width = dimensions
        base_x_scale = 106.0 / length
        base_y_scale = 68.0 / width
        x = x * base_x_scale
        y = y * base_y_scale

        if x < -36:
            x_value = 0
        elif x < -18:
            x_value = 1
        elif x < 0:
            x_value = 2
        elif x < 18:
            x_value = 3
        elif x < 36:
            x_value = 4
        else:
            x_value = 5

        if y < -14:
            y_value = 1
        elif y < 14:
            y_value = 2
        else:
            y_value = 3

        return 3 * x_value + y_value


class KernelTransitionModel(PitchSurfaceModel):
    """Transition model backed by 18-zone pre-computed KDE CSVs.

    Load from a directory containing Area1_Transition.csv ... Area18_Transition.csv.
    """

    def __init__(self, kernel: _ZoneTransitionKernel) -> None:
        self._kernel = kernel

    @classmethod
    def from_csv(cls, data_dir: str | Path) -> KernelTransitionModel:
        kernel = _ZoneTransitionKernel()
        kernel.load_from_csv(data_dir)
        return cls(kernel)

    @property
    def name(self) -> str:
        return "kernel_transition"

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        ball_x = float(state.ball_position[0])
        ball_y = float(state.ball_position[1])

        raw = self._kernel.predict(ball_x, ball_y)
        matrix = np.array(raw, dtype=float)
        m = matrix.max()
        if m > 0:
            matrix /= m

        values = _crop_transition_to_grid(matrix, ball_x, ball_y)

        if values.shape != grid.shape:
            from scipy.ndimage import zoom
            zoom_factors = (
                grid.shape[0] / values.shape[0],
                grid.shape[1] / values.shape[1],
            )
            values = zoom(values, zoom_factors, order=1)

        return PitchSurface(
            values=values,
            grid=grid,
            name=self.name,
            metadata={"ball_position": state.ball_position.tolist()},
        )


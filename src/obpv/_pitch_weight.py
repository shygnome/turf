"""Pitch weight score model: sigmoid-Gaussian heatmap biased toward attacking goal."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
import numpy.typing as npt
from scipy.stats import norm  # type: ignore[import-untyped]

from obpv._base import PitchSurfaceModel
from obpv._grid import PitchGrid
from obpv._state import GameState
from obpv._surface import PitchSurface


def _sigmoid(
    x: npt.NDArray[np.float64], a: float, b: float
) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-(x - b) / a))  # type: ignore[no-any-return]


def generate_heatmap_data(
    field_dimen: tuple[float, float] = (105.0, 68.0),
    n_grid_cells_x: int = 50,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sigmoid-Gaussian pitch weight heatmap, higher near the attacking goal.

    Returns
    -------
    pitch_value : np.ndarray, shape (n_y, n_x)
    xgrid : np.ndarray
    ygrid : np.ndarray
    """
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y

    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2 + dx / 2
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2 + dy / 2

    scale_weights = _sigmoid(xgrid, 30, -15)
    scales = 34 * scale_weights + 34
    x_weights = _sigmoid(xgrid, 30, -15)

    y_center = ygrid[len(ygrid) // 2]
    y_values = np.array(
        [norm.pdf(ygrid, loc=y_center, scale=scale) for scale in scales]
    )

    norm_factors = 1.0 / (scales * sqrt(2 * pi))
    pitch_value = (y_values / norm_factors[:, None]) * x_weights[:, None]

    return pitch_value.T, xgrid, ygrid


class PitchWeightModel(PitchSurfaceModel):
    """Score model based on a sigmoid-Gaussian pitch weight heatmap.

    Higher values near the attacking goal (sigmoid along x) with a Gaussian
    width centred on the pitch (y-axis). Attack direction applied per-frame.
    """

    def __init__(
        self,
        field_dimen: tuple[float, float] = (105.0, 68.0),
        n_grid_cells_x: int = 50,
    ) -> None:
        self.field_dimen = field_dimen
        self.n_grid_cells_x = n_grid_cells_x
        self._base_values, _, _ = generate_heatmap_data(field_dimen, n_grid_cells_x)

    @property
    def name(self) -> str:
        return "pitch_weight"

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        values = self._base_values.copy()

        attacking_direction = state.metadata.get("attacking_direction", "left_to_right")
        if attacking_direction == "right_to_left":
            values = np.fliplr(values)

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
            metadata={
                "frame_id": state.frame_id,
                "attacking_direction": attacking_direction,
            },
        )

    def get_config(self) -> dict[str, object]:
        return {
            "field_dimen": self.field_dimen,
            "n_grid_cells_x": self.n_grid_cells_x,
        }


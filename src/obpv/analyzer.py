"""OBPVAnalyzer: single-frame OBPV computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from obpv._base import PitchSurfaceModel
from obpv._composer import multiply_outputs
from obpv._grid import PitchGrid
from obpv._pitch_control import PPCFParameters, PPCFPitchControlModel
from obpv._pitch_weight import PitchWeightModel
from obpv._state import GameState
from obpv._surface import PitchSurface
from obpv._transition import TransitionGaussModel


@dataclass
class FrameAnalysis:
    """All named PitchSurface layers produced for a single frame."""

    state: GameState
    layers: dict[str, PitchSurface] = field(default_factory=dict)

    def get_layer(self, name: str) -> PitchSurface:
        if name not in self.layers:
            raise KeyError(f"Layer '{name}' not found. Available: {list(self.layers)}")
        return self.layers[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": {
                "frame_id": self.state.frame_id,
                "period": self.state.period,
                "timestamp": self.state.timestamp,
                "ball_position": self.state.ball_position.tolist(),
                "attacking_team_id": self.state.attacking_team_id,
                "defending_team_id": self.state.defending_team_id,
                "metadata": self.state.metadata,
            },
            "layers": {name: s.to_dict() for name, s in self.layers.items()},
        }


class OBPVAnalyzer:
    """Computes Off-Ball Positioning Value (OBPV) for a single tracking frame.

    OBPV = PPCF Ã— Transition Ã— PitchWeight (element-wise product).

    Parameters
    ----------
    ppcf_model:
        Pitch control model.
    transition_model:
        Pass-direction distribution model.
    weight_model:
        Pitch scoring-threat weight model.
    grid:
        Target pitch grid.
    """

    def __init__(
        self,
        ppcf_model: PPCFPitchControlModel,
        transition_model: PitchSurfaceModel,
        weight_model: PitchSurfaceModel,
        grid: PitchGrid,
    ) -> None:
        self._models: dict[str, PitchSurfaceModel] = {
            "ppcf": ppcf_model,
            "transition": transition_model,
            "weight": weight_model,
        }
        self.grid = grid

    def analyze(self, state: GameState) -> FrameAnalysis:
        """Compute OBPV for a single frame and return all component layers."""
        layers: dict[str, PitchSurface] = {}
        for name, model in self._models.items():
            layers[name] = model.compute(state, self.grid)

        composed = np.asarray(multiply_outputs(layers), dtype=float)
        layers["obpv"] = PitchSurface(
            values=composed,
            grid=self.grid,
            name="obpv",
            metadata={"components": list(self._models.keys())},
        )

        return FrameAnalysis(state=state, layers=layers)

    @classmethod
    def default(
        cls,
        grid: PitchGrid | None = None,
        ppcf_params: PPCFParameters | None = None,
        kernel_csv_dir: str | Path | None = None,
        transition_sigma: float = 15.0,
    ) -> OBPVAnalyzer:
        """Build an OBPVAnalyzer with the standard model stack.

        Uses ``KernelTransitionModel`` if *kernel_csv_dir* is provided (requires
        Area1_Transition.csv â€¦ Area18_Transition.csv), otherwise falls back to
        ``TransitionGaussModel`` (sigma=15 m) which requires no data files.
        """
        if grid is None:
            grid = PitchGrid.from_resolution()

        if kernel_csv_dir is not None:
            from obpv._transition import KernelTransitionModel
            transition_model: PitchSurfaceModel = KernelTransitionModel.from_csv(
                kernel_csv_dir
            )
        else:
            transition_model = TransitionGaussModel(sigma=transition_sigma)

        weight_model: PitchSurfaceModel = PitchWeightModel()
        ppcf_model = PPCFPitchControlModel(
            params=ppcf_params,
            apply_offside_check=True,
        )

        return cls(
            ppcf_model=ppcf_model,
            transition_model=transition_model,
            weight_model=weight_model,
            grid=grid,
        )


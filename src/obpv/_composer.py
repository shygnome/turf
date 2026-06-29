"""Composition functions for combining named PitchSurface layers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from obpv._surface import PitchSurface


def multiply_outputs(layers: dict[str, PitchSurface]) -> npt.NDArray[np.float64]:
    """Element-wise product of all layer value arrays.

    OBPV = PPCF x Transition x PitchWeight.
    """
    result = np.ones_like(next(iter(layers.values())).values)
    for surface in layers.values():
        result = result * surface.values
    return result


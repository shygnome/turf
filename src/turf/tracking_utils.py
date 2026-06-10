from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


def player_xy(row: pd.Series, prefix: str) -> tuple[list[float], list[float]]:
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


def ball_xy(row: pd.Series) -> tuple[float | None, float | None]:
    try:
        bx, by = float(row["ball_x"]), float(row["ball_y"])
    except (KeyError, TypeError, ValueError):
        return None, None
    if math.isnan(bx) or math.isnan(by):
        return None, None
    return bx, by

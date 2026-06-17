"""Expected Threat (xT) lookup with bilinear interpolation.

Grid source: Karun Singh (2019) — https://karun.in/blog/expected-threat.html
Data file  : https://karun.in/blog/data/open_xt_12x8_v1.json
8 rows (y, bottom touchline → top) × 12 columns (x, defensive end → attacking end).

Input coordinates must be *normalised* so that positive-x points toward the
opponent's goal (same convention as turf.features.normalize_coords).
Coordinates outside the pitch are clamped to the nearest grid cell.
"""

from __future__ import annotations

__all__ = ["xt_at", "xt_gain"]

# ── Grid constants ────────────────────────────────────────────────────────────
_ROWS: int = 8
_COLS: int = 12
_HL: float = 52.5   # half pitch length
_HW: float = 34.0   # half pitch width

# Cell dimensions in metres
_DX: float = 2 * _HL / _COLS   # 8.75 m per column
_DY: float = 2 * _HW / _ROWS   # 8.50 m per row

# ── Published xT values (Karun Singh, 2019 — open_xt_12x8_v1.json) ───────────
# Indexed [row][col]: row 0 = bottom touchline, col 0 = defensive end.
# Grid is symmetric in y: rows 0↔7, 1↔6, 2↔5, 3↔4.
_XT_GRID: list[list[float]] = [
    # row 0 — bottom touchline flank
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344,
     0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590],
    # row 1
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540,
     0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    # row 2
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128,
     0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    # row 3 — centre-ish (mirror of row 4)
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598,
     0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    # row 4 — centre-ish (mirror of row 3)
    [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598,
     0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
    # row 5 (mirror of row 2)
    [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128,
     0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
    # row 6 (mirror of row 1)
    [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540,
     0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
    # row 7 — top touchline flank (mirror of row 0)
    [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344,
     0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590],
]


def _bilinear(col_f: float, row_f: float) -> float:
    """Bilinear interpolation over _XT_GRID at fractional grid indices."""
    col_f = max(0.0, min(_COLS - 1, col_f))
    row_f = max(0.0, min(_ROWS - 1, row_f))

    c0 = max(0, min(_COLS - 2, int(col_f)))
    c1 = c0 + 1
    r0 = max(0, min(_ROWS - 2, int(row_f)))
    r1 = r0 + 1

    alpha = col_f - c0   # fractional x weight toward c1
    beta = row_f - r0    # fractional y weight toward r1

    v00 = _XT_GRID[r0][c0]
    v01 = _XT_GRID[r0][c1]
    v10 = _XT_GRID[r1][c0]
    v11 = _XT_GRID[r1][c1]

    return (
        (1 - beta) * ((1 - alpha) * v00 + alpha * v01)
        + beta * ((1 - alpha) * v10 + alpha * v11)
    )


def xt_at(x: float, y: float) -> float:
    """Return the xT value at normalised pitch coordinates (x, y).

    Positive x points toward the opponent's goal.
    Coordinates outside the pitch are clamped to the nearest grid cell.
    """
    col_f = (x + _HL) / _DX - 0.5
    row_f = (y + _HW) / _DY - 0.5
    return _bilinear(col_f, row_f)


def xt_gain(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    completed: bool,
) -> float:
    """Return xT gain for a pass.

    For incomplete passes, the gain is 0 (possession lost, no threat created).
    All coordinates must be normalised (positive-x = attacking direction).
    """
    if not completed:
        return 0.0
    return xt_at(end_x, end_y) - xt_at(start_x, start_y)

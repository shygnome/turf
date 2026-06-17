"""Expected Threat (xT) lookup with bilinear interpolation.

Grid source: Karun Singh (2019) — https://karun.in/blog/expected-threat.html
12 rows (y, bottom touchline → top) × 16 columns (x, defensive end → attacking end).

Input coordinates must be *normalised* so that positive-x points toward the
opponent's goal (same convention as turf.features.normalize_coords).
Coordinates outside the pitch are clamped to the nearest grid cell.
"""

from __future__ import annotations

__all__ = ["xt_at", "xt_gain"]

# ── Grid constants ────────────────────────────────────────────────────────────
_ROWS: int = 12
_COLS: int = 16
_HL: float = 52.5   # half pitch length
_HW: float = 34.0   # half pitch width

# Cell dimensions in metres
_DX: float = 2 * _HL / _COLS   # 6.5625 m per column
_DY: float = 2 * _HW / _ROWS   # 5.6667 m per row

# ── Published xT values (Karun Singh, 2019) ───────────────────────────────────
# Indexed [row][col]: row 0 = bottom touchline, col 0 = defensive end.
# The grid is symmetric in y (rows 0↔11, 1↔10, …, 5↔6).
_XT_GRID: list[list[float]] = [
    # row 0 — bottom touchline flank
    [0.00138696, 0.00179652, 0.00198845, 0.00230054, 0.00341642, 0.00480685,
     0.00648072, 0.00878827, 0.00983276, 0.01250541, 0.01641546, 0.02199006,
     0.02476991, 0.02832374, 0.03749616, 0.07688630],
    # row 1
    [0.00149581, 0.00200948, 0.00242483, 0.00343019, 0.00547537, 0.00765884,
     0.01059050, 0.01430664, 0.01685016, 0.02151959, 0.02913682, 0.04037745,
     0.04747635, 0.05966177, 0.07643827, 0.16871366],
    # row 2
    [0.00180826, 0.00262610, 0.00329935, 0.00473327, 0.00753498, 0.01036048,
     0.01410312, 0.01890769, 0.02264516, 0.02933567, 0.03921951, 0.05456905,
     0.06506226, 0.08064506, 0.10827274, 0.22487759],
    # row 3
    [0.00199848, 0.00316407, 0.00414595, 0.00590716, 0.00929847, 0.01291208,
     0.01813444, 0.02503358, 0.03003517, 0.03931578, 0.05337626, 0.07532834,
     0.09091617, 0.11527757, 0.16175527, 0.31983073],
    # row 4
    [0.00216297, 0.00345648, 0.00479167, 0.00702476, 0.01109063, 0.01567898,
     0.02253129, 0.03165963, 0.03819117, 0.05027395, 0.06918127, 0.09879266,
     0.12058764, 0.15730840, 0.22447669, 0.45092527],
    # row 5 — centre
    [0.00211785, 0.00349927, 0.00502978, 0.00754004, 0.01194432, 0.01725565,
     0.02528898, 0.03616068, 0.04370499, 0.05862978, 0.08161558, 0.11765069,
     0.14445399, 0.19062878, 0.27699558, 0.56090170],
    # row 6 — centre (mirror of row 5)
    [0.00211785, 0.00349927, 0.00502978, 0.00754004, 0.01194432, 0.01725565,
     0.02528898, 0.03616068, 0.04370499, 0.05862978, 0.08161558, 0.11765069,
     0.14445399, 0.19062878, 0.27699558, 0.56090170],
    # row 7 (mirror of row 4)
    [0.00216297, 0.00345648, 0.00479167, 0.00702476, 0.01109063, 0.01567898,
     0.02253129, 0.03165963, 0.03819117, 0.05027395, 0.06918127, 0.09879266,
     0.12058764, 0.15730840, 0.22447669, 0.45092527],
    # row 8 (mirror of row 3)
    [0.00199848, 0.00316407, 0.00414595, 0.00590716, 0.00929847, 0.01291208,
     0.01813444, 0.02503358, 0.03003517, 0.03931578, 0.05337626, 0.07532834,
     0.09091617, 0.11527757, 0.16175527, 0.31983073],
    # row 9 (mirror of row 2)
    [0.00180826, 0.00262610, 0.00329935, 0.00473327, 0.00753498, 0.01036048,
     0.01410312, 0.01890769, 0.02264516, 0.02933567, 0.03921951, 0.05456905,
     0.06506226, 0.08064506, 0.10827274, 0.22487759],
    # row 10 (mirror of row 1)
    [0.00149581, 0.00200948, 0.00242483, 0.00343019, 0.00547537, 0.00765884,
     0.01059050, 0.01430664, 0.01685016, 0.02151959, 0.02913682, 0.04037745,
     0.04747635, 0.05966177, 0.07643827, 0.16871366],
    # row 11 — top touchline flank (mirror of row 0)
    [0.00138696, 0.00179652, 0.00198845, 0.00230054, 0.00341642, 0.00480685,
     0.00648072, 0.00878827, 0.00983276, 0.01250541, 0.01641546, 0.02199006,
     0.02476991, 0.02832374, 0.03749616, 0.07688630],
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
    Coordinates outside the pitch are clamped to the nearest cell.
    """
    # Map to fractional grid column / row (cell-centre aligned)
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

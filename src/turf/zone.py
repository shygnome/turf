"""Pitch zone assignment for three tactical zone schemes.

All coordinates must be in SkillCorner space (metres, origin at pitch centre):
  x ∈ [-52.5, 52.5]  goal-to-goal   (negative x = own defensive end)
  y ∈ [-34.0, 34.0]  touchline-to-touchline

Boundary convention: points exactly on a boundary fall into the zone to their
right / above (half-open intervals [left, right)).

Zone ID quick reference
-----------------------
Simple Thirds (3):
  t_def  t_mid  t_att

Van Gaal 6×3 (18):
  vg_{col}_{row}
  col ∈ {d_box, d_mid, m_def, m_att, a_mid, a_box}   (x, defensive → attacking)
  row ∈ {fl, ctr, fr}                        (y, bottom flank / centre / top flank)

Guardiola (20):
  pg_gk_def  pg_gk_att                                (GK boxes, span inner lanes)
  pg_{lane}_{col}   lane ∈ {lf, rf}                   (flank lanes, 6 col zones each)
  pg_{lane}_{half}  lane ∈ {lhs, ctr, rhs}            (inner lanes, half ∈ {def, att})
"""

from __future__ import annotations

import bisect
from enum import Enum

__all__ = [
    "ZoneScheme",
    "assign_zone",
    "GUARDIOLA_ZONE_NUMBER",
    "GUARDIOLA_ZONE_BOUNDS",
]


# ── Guardiola zone metadata ───────────────────────────────────────────────────
# Numbers match the draw_zones.py visual: z1-6 = L-Flank, z7-12 = R-Flank,
# z13-14 = L-HalfSpace, z15-16 = Center, z17-18 = R-HalfSpace, z19-20 = GK boxes.

GUARDIOLA_ZONE_NUMBER: dict[str, int] = {
    # L-Flank (y < -20.16), columns def → att
    "pg_lf_d_box": 1,   "pg_lf_d_mid": 2,  "pg_lf_m_def": 3,
    "pg_lf_m_att": 4,   "pg_lf_a_mid": 5,  "pg_lf_a_box": 6,
    # R-Flank (y > +20.16), columns def → att
    "pg_rf_d_box": 7,   "pg_rf_d_mid": 8,  "pg_rf_m_def": 9,
    "pg_rf_m_att": 10,  "pg_rf_a_mid": 11, "pg_rf_a_box": 12,
    # Inner lanes, def half then att half
    "pg_lhs_def": 13,   "pg_lhs_att": 14,
    "pg_ctr_def": 15,   "pg_ctr_att": 16,
    "pg_rhs_def": 17,   "pg_rhs_att": 18,
    # GK boxes (span all inner lanes)
    "pg_gk_def": 19,    "pg_gk_att": 20,
}

# (x_min, x_max, y_min, y_max) in SkillCorner metres for each zone.
GUARDIOLA_ZONE_BOUNDS: dict[str, tuple[float, float, float, float]] = {
    # L-Flank
    "pg_lf_d_box": (-52.5, -36.0, -34.0,  -20.16),
    "pg_lf_d_mid": (-36.0, -18.0, -34.0,  -20.16),
    "pg_lf_m_def": (-18.0,   0.0, -34.0,  -20.16),
    "pg_lf_m_att": (  0.0,  18.0, -34.0,  -20.16),
    "pg_lf_a_mid": ( 18.0,  36.0, -34.0,  -20.16),
    "pg_lf_a_box": ( 36.0,  52.5, -34.0,  -20.16),
    # R-Flank
    "pg_rf_d_box": (-52.5, -36.0,  20.16,  34.0),
    "pg_rf_d_mid": (-36.0, -18.0,  20.16,  34.0),
    "pg_rf_m_def": (-18.0,   0.0,  20.16,  34.0),
    "pg_rf_m_att": (  0.0,  18.0,  20.16,  34.0),
    "pg_rf_a_mid": ( 18.0,  36.0,  20.16,  34.0),
    "pg_rf_a_box": ( 36.0,  52.5,  20.16,  34.0),
    # L-HalfSpace
    "pg_lhs_def":  (-36.0,   0.0, -20.16,  -9.15),
    "pg_lhs_att":  (  0.0,  36.0, -20.16,  -9.15),
    # Center
    "pg_ctr_def":  (-36.0,   0.0,  -9.15,   9.15),
    "pg_ctr_att":  (  0.0,  36.0,  -9.15,   9.15),
    # R-HalfSpace
    "pg_rhs_def":  (-36.0,   0.0,   9.15,  20.16),
    "pg_rhs_att":  (  0.0,  36.0,   9.15,  20.16),
    # GK boxes — span all inner lanes in y
    "pg_gk_def":   (-52.5, -36.0, -20.16,  20.16),
    "pg_gk_att":   ( 36.0,  52.5, -20.16,  20.16),
}


class ZoneScheme(Enum):
    THIRDS = "thirds"
    VAN_GAAL = "van_gaal"
    GUARDIOLA = "guardiola"


# ── Pitch constants (metres, SkillCorner) ─────────────────────────────────────
_HL: float = 52.5  # half pitch length
_HW: float = 34.0  # half pitch width
_PEN_DEPTH: float = 16.5  # penalty area depth from goal line
_PEN_HALF_W: float = 20.16  # half penalty area width  (= 40.32 / 2)
_KC_R: float = 9.15  # kickoff-circle radius = half-space y-boundary
_INNER: float = 18.0  # inner column width    (= (105 - 2×16.5) / 4)

# ── Van Gaal boundaries ───────────────────────────────────────────────────────
_VG_X: list[float] = [
    -_HL,
    -_HL + _PEN_DEPTH,
    -_INNER,
    0.0,
    _INNER,
    _HL - _PEN_DEPTH,
    _HL,
]
_VG_X_NAMES: list[str] = ["d_box", "d_mid", "m_def", "m_att", "a_mid", "a_box"]

_VG_Y: list[float] = [-_HW, -_PEN_HALF_W, _PEN_HALF_W, _HW]
_VG_Y_NAMES: list[str] = ["fl", "ctr", "fr"]

# ── Guardiola boundaries ──────────────────────────────────────────────────────
_PG_LANE_Y: list[float] = [-_HW, -_PEN_HALF_W, -_KC_R, _KC_R, _PEN_HALF_W, _HW]
_PG_LANE_NAMES: list[str] = ["lf", "lhs", "ctr", "rhs", "rf"]

# Inner lanes (lhs/ctr/rhs) use only penalty-depth and halfway as x-cuts.
# Points outside [-36, 36) on inner lanes → GK box zones instead.
_PG_INNER_X: list[float] = [-(_HL - _PEN_DEPTH), 0.0, _HL - _PEN_DEPTH]
_PG_INNER_NAMES: list[str] = ["def", "att"]
_PG_GK_X: float = _HL - _PEN_DEPTH  # 36.0


def _bucket(val: float, bounds: list[float]) -> int:
    """0-based index of the interval [bounds[i], bounds[i+1]) containing val."""
    idx = bisect.bisect_right(bounds, val) - 1
    return max(0, min(idx, len(bounds) - 2))


def _thirds(x: float) -> str:
    names = ("t_def", "t_mid", "t_att")
    return names[_bucket(x, [-_HL, -_HL / 3, _HL / 3, _HL])]


def _van_gaal(x: float, y: float) -> str:
    col = _VG_X_NAMES[_bucket(x, _VG_X)]
    row = _VG_Y_NAMES[_bucket(y, _VG_Y)]
    return f"vg_{col}_{row}"


def _guardiola(x: float, y: float) -> str:
    lane_idx = _bucket(y, _PG_LANE_Y)
    lane = _PG_LANE_NAMES[lane_idx]

    if lane_idx in (1, 2, 3):  # lhs / ctr / rhs — check GK box first
        if x < -_PG_GK_X:
            return "pg_gk_def"
        if x >= _PG_GK_X:
            return "pg_gk_att"
        half = _PG_INNER_NAMES[_bucket(x, _PG_INNER_X)]
        return f"pg_{lane}_{half}"

    # Flank lanes (lf / rf) — 6 column zones same as Van Gaal
    col = _VG_X_NAMES[_bucket(x, _VG_X)]
    return f"pg_{lane}_{col}"


def assign_zone(x: float, y: float, scheme: ZoneScheme) -> str:
    """Zone label for a point (x, y) under *scheme*.

    See module docstring for coordinate system and zone ID reference.
    """
    if scheme is ZoneScheme.THIRDS:
        return _thirds(x)
    if scheme is ZoneScheme.VAN_GAAL:
        return _van_gaal(x, y)
    if scheme is ZoneScheme.GUARDIOLA:
        return _guardiola(x, y)
    raise ValueError(f"Unknown scheme: {scheme}")

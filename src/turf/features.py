"""Pass covariate extraction for the LBP evaluation pipeline.

Input  : labeled_metadata.csv (from LEAK label-pass), wc2022_goals.csv
Output : pass_features.csv — one row per labeled pass with zone, distance,
         angle, game-state, territory-entry, and outcome covariates.

Coordinates are in SkillCorner space (metres, origin at pitch centre):
  x ∈ [-52.5, 52.5]  goal-to-goal
  y ∈ [-34.0, 34.0]  touchline-to-touchline

All spatial features are computed after normalizing to the passer's attack
direction so that positive-x always points toward the opponent's goal.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from turf.xt import xt_gain as _xt_gain
from turf.zone import ZoneScheme, assign_zone

__all__ = [
    "goal_to_period_seconds",
    "build_score_timeline",
    "score_at",
    "score_diff",
    "pass_distance",
    "pass_angle",
    "territory_entry",
    "normalize_coords",
    "build_attack_dir_cache",
    "extract_pass_features",
]

_HL: float = 52.5  # half pitch length (metres)

# Period start minutes (broadcast, 1-indexed cumulative)
_PERIOD_START_MINUTE: dict[int, int] = {1: 1, 2: 46, 3: 91, 4: 106}


# ── Game-state helpers ────────────────────────────────────────────────────────


def goal_to_period_seconds(period: int, minute: int, added_time: int) -> float:
    """Convert a broadcast goal time to seconds from that period's kick-off."""
    offset = _PERIOD_START_MINUTE[period]
    return (minute - offset) * 60.0 + added_time * 60.0


def build_score_timeline(
    goals_df: pd.DataFrame,
) -> dict[int, list[tuple[str, int, float]]]:
    """Return {match_id: [(scoring_team, period, period_seconds), ...]} sorted by time.
    """
    timeline: dict[int, list[tuple[str, int, float]]] = {}
    for _, row in goals_df.iterrows():
        mid = int(row["match_id"])
        t = goal_to_period_seconds(
            int(row["period"]), int(row["minute"]), int(row["added_time"])
        )
        timeline.setdefault(mid, []).append(
            (str(row["scoring_team"]), int(row["period"]), t)
        )
    for mid in timeline:
        timeline[mid].sort(key=lambda e: (e[1], e[2]))
    return timeline


def score_at(
    timeline: dict[int, list[tuple[str, int, float]]],
    match_id: int,
    period: int,
    time_s: float,
) -> tuple[int, int]:
    """Return (home_score, away_score) at (period, time_s).

    Goals scored at exactly time_s are not counted (pass happened before them).
    """
    home, away = 0, 0
    for scoring_team, g_period, g_time in timeline.get(match_id, []):
        if g_period < period or (g_period == period and g_time < time_s):
            if scoring_team == "Home":
                home += 1
            else:
                away += 1
    return home, away


def score_diff(
    timeline: dict[int, list[tuple[str, int, float]]],
    match_id: int,
    period: int,
    time_s: float,
    passer_team: str,
) -> int:
    """Score difference from the passer's perspective (passer_score − opponent)."""
    home, away = score_at(timeline, match_id, period, time_s)
    return (home - away) if passer_team == "Home" else (away - home)


# ── Spatial helpers ───────────────────────────────────────────────────────────


def normalize_coords(x: float, y: float, attack_dir: int) -> tuple[float, float]:
    """Flip coordinates so positive-x always points toward the opponent's goal."""
    return x * attack_dir, y * attack_dir


def pass_distance(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float:
    """Euclidean pass distance in metres (raw, un-normalized coordinates)."""
    return math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)


def pass_angle(
    start_x: float, start_y: float, end_x: float, end_y: float
) -> float:
    """Pass angle in radians relative to the positive-x axis, range (−π, π].

    Call with normalized coordinates so 0 = straight forward toward goal.
    """
    return math.atan2(end_y - start_y, end_x - start_x)


def territory_entry(norm_end_x: float) -> bool:
    """True if the pass lands in the attacking final third (norm_end_x ≥ HL/3)."""
    return norm_end_x >= _HL / 3


# ── Attack-direction cache ────────────────────────────────────────────────────


def build_attack_dir_cache(
    pass_dir: Path,
    labeled_df: pd.DataFrame,
) -> dict[tuple[int, str], int]:
    """Return {(period, team): attack_direction} for all discoverable combos.

    Scans available lines.csv files in *pass_dir* and calls
    detect_attack_direction from the LEAK module. Combos with no lines.csv
    are omitted; callers should skip rows with missing keys.
    """
    from leak.pass_label import detect_attack_direction  # noqa: PLC0415

    cache: dict[tuple[int, str], int] = {}
    for _, row in labeled_df.iterrows():
        period = int(row["period"])
        passer_team = str(row["team"])
        key = (period, passer_team)
        if key in cache:
            continue

        defending_team = "Away" if passer_team == "Home" else "Home"
        lines_path = pass_dir / str(int(row["event_idx"])) / "lines.csv"
        if not lines_path.exists():
            continue

        try:
            lines_df = pd.read_csv(lines_path)
            cache[key] = detect_attack_direction(lines_df, defending_team)
        except (ValueError, KeyError):
            continue

    return cache


# ── Main extraction function ──────────────────────────────────────────────────


def extract_pass_features(
    labeled_df: pd.DataFrame,
    goals_timeline: dict[int, list[tuple[str, int, float]]],
    attack_dirs: dict[tuple[int, str], int],
    match_id: int,
) -> pd.DataFrame:
    """Extract covariates for all labeled passes in *labeled_df*.

    Rows whose (period, team) combination is absent from *attack_dirs* are
    silently skipped (attack direction could not be determined).

    Returns a DataFrame with one row per included pass.
    """
    rows: list[dict[str, object]] = []
    for _, row in labeled_df.iterrows():
        period = int(row["period"])
        team = str(row["team"])
        attack_dir = attack_dirs.get((period, team))
        if attack_dir is None:
            continue

        time_s = float(row["start_time"])
        sx, sy = float(row["start_x"]), float(row["start_y"])
        ex = float(row["inferred_end_x"])
        ey = float(row["inferred_end_y"])

        norm_sx, norm_sy = normalize_coords(sx, sy, attack_dir)
        norm_ex, norm_ey = normalize_coords(ex, ey, attack_dir)

        rows.append(
            {
                "event_idx": int(row["event_idx"]),
                "match_id": match_id,
                "team": team,
                "period": period,
                "start_time": time_s,
                "is_line_breaking": bool(row["is_line_breaking"]),
                "lines_broken_count": int(row["lines_broken_count"]),
                "pass_outcome": str(row["subtype"]) == "success",
                "zone_thirds": assign_zone(norm_sx, norm_sy, ZoneScheme.THIRDS),
                "zone_van_gaal": assign_zone(norm_sx, norm_sy, ZoneScheme.VAN_GAAL),
                "zone_guardiola": assign_zone(norm_sx, norm_sy, ZoneScheme.GUARDIOLA),
                "pass_distance": pass_distance(sx, sy, ex, ey),
                "pass_angle": pass_angle(norm_sx, norm_sy, norm_ex, norm_ey),
                "territory_entry": territory_entry(norm_ex),
                "score_diff": score_diff(
                    goals_timeline, match_id, period, time_s, team
                ),
                "xt_gain": _xt_gain(
                    norm_sx, norm_sy, norm_ex, norm_ey,
                    completed=str(row["subtype"]) == "success",
                ),
            }
        )

    return pd.DataFrame(rows)

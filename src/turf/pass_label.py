from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]


def detect_attack_direction(lines_df: pd.DataFrame, defending_team: str) -> int:
    """
    Return +1 or -1: the x-direction of attack (toward the defending GK's goal).

    Uses the median x of the defending GK across all frames. GK is identified
    by the player column whose _line value is 0. If GK x > 0, the defending goal
    is on the positive side, so the attack runs in the +x direction.
    """
    line_cols = [
        c
        for c in lines_df.columns
        if c.startswith(defending_team + "_") and c.endswith("_line")
    ]
    for col in line_cols:
        if (lines_df[col] == 0).any():
            player_id = col[: -len("_line")].rsplit("_", 1)[1]
            gk_x_col = f"{defending_team}_{player_id}_x"
            if gk_x_col not in lines_df.columns:
                continue
            median_x: float = float(lines_df[gk_x_col].median())
            return 1 if median_x > 0 else -1
    raise ValueError(
        f"No GK (line=0) found for team '{defending_team}' in lines DataFrame"
    )


def compute_conceptual_line_xs(
    frame_row: pd.Series,
    defending_team: str,
    attack_dir: int,
) -> list[float]:
    """
    Return x-positions of conceptual lines [L1_x, L2_x, ..., Ln_x] from a single
    frame row of lines.csv, sorted from most-advanced (L1) to deepest (Ln).

    L1 = first line encountered in the attack direction (the pressing forwards).
    Conceptual order is determined by sorting LEAK line means by ascending norm
    (x * attack_dir), so L1 always has the smallest norm regardless of direction.

    GK (line=0) is excluded.
    """
    line_cols = [
        c
        for c in frame_row.index
        if c.startswith(defending_team + "_") and c.endswith("_line")
    ]

    line_xs: dict[int, list[float]] = {}
    for col in line_cols:
        line_val = frame_row[col]
        if pd.isna(line_val) or int(line_val) == 0:
            continue
        leak_line = int(line_val)
        player_id = col[: -len("_line")].rsplit("_", 1)[1]
        x_col = f"{defending_team}_{player_id}_x"
        if x_col not in frame_row.index:
            continue
        x_val = frame_row[x_col]
        if pd.isna(x_val):
            continue
        line_xs.setdefault(leak_line, []).append(float(x_val))

    if not line_xs:
        return []

    line_means = {k: sum(v) / len(v) for k, v in line_xs.items()}
    # Sort by ascending norm: smallest norm = most advanced (first encountered)
    return sorted(line_means.values(), key=lambda x: x * attack_dir)


def assign_zone(
    ball_x: float,
    conceptual_line_xs: list[float],
    attack_dir: int,
) -> str:
    """
    Map a ball x-position to a zone name given conceptual line positions.

    conceptual_line_xs: [L1_x, L2_x, ..., Ln_x] in conceptual order
                        (L1 = most advanced press, already sorted by ascending norm).
    attack_dir: +1 or -1.

    Zones:
      L1-zone   — ball has not yet beaten the most advanced press
      L2-zone   — ball has beaten L1 but not L2
      ...
      Ln-zone   — ball has beaten L(n-1) but not Ln
      danger-zone — ball has beaten all outfield lines (between last line and GK)
    """
    norm_ball = ball_x * attack_dir
    for i, line_x in enumerate(conceptual_line_xs):
        if norm_ball < line_x * attack_dir:
            return f"L{i + 1}-zone"
    return "danger-zone"


def _zone_index(zone: str, n_lines: int) -> int:
    """Convert zone name to a comparable integer index.

    L1-zone=0, L2-zone=1, ..., Ln-zone=n-1, danger-zone=n.
    """
    if zone == "danger-zone":
        return n_lines
    return int(zone[1 : zone.index("-")]) - 1


def detect_line_break(
    start_ball_x: float,
    end_ball_x: float,
    start_line_xs: list[float],
    end_line_xs: list[float],
    attack_dir: int,
) -> dict[str, Any]:
    """
    Determine whether a pass broke through one or more defensive unit lines.

    start_line_xs: conceptual line x-positions at the start frame of the event.
    end_line_xs:   conceptual line x-positions at the end frame of the event.
    Both lists must be in conceptual order (L1 first, as returned by
    compute_conceptual_line_xs).

    Returns a dict with:
      is_line_breaking  — True if the ball advanced through at least one line
      lines_broken_count — number of conceptual lines crossed (0 if not line breaking)
      lines_broken       — list of conceptual line numbers crossed, e.g. [1, 2]
    """
    start_zone = assign_zone(start_ball_x, start_line_xs, attack_dir)
    end_zone = assign_zone(end_ball_x, end_line_xs, attack_dir)

    start_idx = _zone_index(start_zone, len(start_line_xs))
    end_idx = _zone_index(end_zone, len(end_line_xs))

    if end_idx <= start_idx:
        return {"is_line_breaking": False, "lines_broken_count": 0, "lines_broken": []}

    lines_broken = list(range(start_idx + 1, end_idx + 1))
    return {
        "is_line_breaking": True,
        "lines_broken_count": len(lines_broken),
        "lines_broken": lines_broken,
    }


def label_event(event_dir: Path, defending_team: str) -> dict[str, Any]:
    """
    Compute line-break labels for a single pass event.

    Reads lines.csv from event_dir. Uses the first frame for start line positions
    and ball position, and the last frame for end line positions and ball position.

    Returns a dict suitable for merging into metadata.
    """
    lines_path = event_dir / "lines.csv"
    lines_df = pd.read_csv(lines_path)

    attack_dir = detect_attack_direction(lines_df, defending_team)

    first_row = lines_df.iloc[0]
    last_row = lines_df.iloc[-1]

    start_line_xs = compute_conceptual_line_xs(first_row, defending_team, attack_dir)
    end_line_xs = compute_conceptual_line_xs(last_row, defending_team, attack_dir)

    start_ball_x = float(first_row["ball_x"])
    end_ball_x = float(last_row["ball_x"])

    return detect_line_break(
        start_ball_x, end_ball_x, start_line_xs, end_line_xs, attack_dir
    )


def label_all_passes(
    output_dir: Path,
    dataset_id: str,
    match_id: str,
) -> pd.DataFrame:
    """
    Label all pass events for a match and return the labeled DataFrame.

    Reads from output_dir/<dataset_id>/<match_id>/pass/metadata.csv and the
    per-event lines.csv files produced by `turf analyze leak extract-line`.

    Writes labeled_metadata.csv next to metadata.csv and returns the DataFrame.
    """
    pass_dir = output_dir / dataset_id / match_id / "pass"
    meta_path = pass_dir / "metadata.csv"
    metadata = pd.read_csv(meta_path)

    label_rows: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        event_idx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = pass_dir / str(event_idx)
        lines_path = event_dir / "lines.csv"
        if not lines_path.exists():
            label_rows.append(
                {
                    "event_idx": event_idx,
                    "is_line_breaking": None,
                    "lines_broken_count": None,
                    "lines_broken": None,
                }
            )
            continue

        try:
            labels = label_event(event_dir, defending_team)
        except (ValueError, KeyError):
            label_rows.append(
                {
                    "event_idx": event_idx,
                    "is_line_breaking": None,
                    "lines_broken_count": None,
                    "lines_broken": None,
                }
            )
            continue

        label_rows.append(
            {
                "event_idx": event_idx,
                "is_line_breaking": labels["is_line_breaking"],
                "lines_broken_count": labels["lines_broken_count"],
                "lines_broken": str(labels["lines_broken"]),
            }
        )

    labels_df = pd.DataFrame(label_rows)
    labeled = metadata.merge(labels_df, on="event_idx", how="left")
    labeled.to_csv(pass_dir / "labeled_metadata.csv", index=False)
    return labeled


def compute_pass_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics from a labeled_metadata DataFrame.

    Rows where is_line_breaking is NaN/None are treated as skipped (no lines.csv).
    by_team and by_period totals count only labeled (non-skipped) rows.
    """
    total = len(df)
    labeled_mask = df["is_line_breaking"].notna()
    n_labeled = int(labeled_mask.sum())
    n_skipped = total - n_labeled

    labeled = df[labeled_mask]
    n_breaking = int(labeled["is_line_breaking"].eq(True).sum())
    n_not_breaking = n_labeled - n_breaking

    breaking = labeled[labeled["is_line_breaking"].eq(True)]
    lines_broken_dist: dict[int, int] = {}
    if not breaking.empty:
        for k, cnt in breaking["lines_broken_count"].value_counts().items():
            lines_broken_dist[int(k)] = int(cnt)

    def _group_stats(col: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for key in sorted(labeled[col].dropna().unique()):
            mask = labeled[col] == key
            result[str(key)] = {
                "breaking": int(labeled[mask]["is_line_breaking"].eq(True).sum()),
                "total": int(mask.sum()),
            }
        return result

    return {
        "total": total,
        "n_labeled": n_labeled,
        "n_skipped": n_skipped,
        "n_breaking": n_breaking,
        "n_not_breaking": n_not_breaking,
        "lines_broken_dist": lines_broken_dist,
        "by_team": _group_stats("team"),
        "by_period": {int(k): v for k, v in _group_stats("period").items()},
        "by_subtype": _group_stats("subtype"),
    }

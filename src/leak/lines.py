from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]


def find_goalkeeper(df: pd.DataFrame, team: str) -> str:
    """Return player number of the goalkeeper (max abs-x position at first frame)."""
    x_cols = [c for c in df.columns if c.startswith(team + "_") and c.endswith("_x")]
    valid = df.iloc[0][x_cols].dropna()
    if valid.empty:
        raise ValueError(f"No {team} player positions in first frame")
    gk_col: str = valid.abs().idxmax()
    # Away_3_x -> rsplit("_x") -> "Away_3" -> rsplit("_", 1) -> "3"
    return gk_col.rsplit("_x", 1)[0].rsplit("_", 1)[1]


def _calinski_harabasz(
    x: npt.NDArray[np.float64], labels: npt.NDArray[np.intp]
) -> float:
    """CH index for 1-D data; 0.0 when undefined (k<2 or k>=n)."""
    n = len(x)
    clusters = np.unique(labels)
    k = len(clusters)
    if k < 2 or k >= n:
        return 0.0
    overall_mean = x.mean()
    ssb = sum(
        float((labels == c).sum()) * float((x[labels == c].mean() - overall_mean) ** 2)
        for c in clusters
    )
    ssw = sum(
        float(((x[labels == c] - x[labels == c].mean()) ** 2).sum()) for c in clusters
    )
    if ssw < 1e-12:
        return float("inf")
    return (ssb / (k - 1)) / (ssw / (n - k))


def _cluster(x: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.intp]:
    """Ward agglomerative clustering; returns 0-indexed labels."""
    if k == 1:
        return np.zeros(len(x), dtype=np.intp)
    Z: Any = linkage(x.reshape(-1, 1), method="ward")
    result: npt.NDArray[np.intp] = (
        fcluster(Z, t=k, criterion="maxclust").astype(np.intp) - 1
    )
    return result


def _merge_singletons(
    x: npt.NDArray[np.float64], labels: npt.NDArray[np.intp], min_size: int
) -> npt.NDArray[np.intp]:
    """Merge clusters smaller than min_size into the nearest cluster by mean x."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique: npt.NDArray[np.intp]
        counts: npt.NDArray[np.intp]
        unique, counts = np.unique(labels, return_counts=True)
        for c, cnt in zip(unique, counts, strict=False):
            if cnt < min_size:
                c_mean = float(x[labels == c].mean())
                others = [o for o in unique if o != c]
                if not others:
                    break

                def _key(o: np.intp, cm: float = c_mean) -> float:
                    return abs(float(x[labels == o].mean()) - cm)

                nearest = min(others, key=_key)
                labels[labels == c] = nearest
                changed = True
                break  # restart after each merge
    return labels


def detect_lines_frame(
    x_positions: dict[str, float],
    min_per_line: int = 2,
    min_lines: int = 2,
    max_lines: int = 4,
) -> dict[str, int]:
    """
    Detect defensive unit lines for outfield players in a single frame.

    x_positions: {player_number -> x coordinate} — GK already excluded
    Returns: {player_number -> line_number} where 1 = deepest (lowest x)
    """
    if not x_positions:
        return {}

    players = list(x_positions.keys())
    x: npt.NDArray[np.float64] = np.array(
        [x_positions[p] for p in players], dtype=np.float64
    )
    n = len(players)

    if n == 1:
        return {players[0]: 1}

    lo = max(1, min(min_lines, n // min_per_line))
    hi = min(max_lines, n // min_per_line)

    best_labels: npt.NDArray[np.intp] | None = None
    best_score = -1.0

    for k in range(lo, hi + 1):
        labels = _cluster(x, k)
        unique: npt.NDArray[np.intp]
        counts: npt.NDArray[np.intp]
        unique, counts = np.unique(labels, return_counts=True)
        valid = (
            len(unique) == k
            and len(unique) >= min_lines
            and bool(np.all(counts >= min_per_line))
        )
        if valid:
            score = _calinski_harabasz(x, labels)
            if score > best_score:
                best_score = score
                best_labels = labels.copy()

    if best_labels is None:
        # Fallback: no clustering met all constraints; use smallest allowed k
        # and merge any undersized groups afterward
        k = max(1, min(min_lines, max(1, n // min_per_line)))
        best_labels = _cluster(x, k)
        best_labels = _merge_singletons(x, best_labels, min_per_line)

    # Remap to 1-indexed labels ordered by mean x (1 = deepest)
    unique_clusters = np.unique(best_labels)
    assert best_labels is not None
    ordered = sorted(unique_clusters, key=lambda c: float(x[best_labels == c].mean()))
    remap = {int(old): new + 1 for new, old in enumerate(ordered)}

    return {p: remap[int(best_labels[i])] for i, p in enumerate(players)}


def analyze_lines(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Add unit line assignments to every frame in df.

    Adds {team}_{player}_line columns: 0 for GK, 1-4 for outfield, NaN if absent.
    Also adds line_count column with the number of lines detected per frame.
    """
    result: pd.DataFrame = df.copy()
    gk_num = find_goalkeeper(df, team)
    gk_x_col = f"{team}_{gk_num}_x"

    x_cols = [
        c
        for c in df.columns
        if c.startswith(team + "_") and c.endswith("_x") and c != gk_x_col
    ]
    player_nums = [c.rsplit("_x", 1)[0].rsplit("_", 1)[1] for c in x_cols]

    result[f"{team}_{gk_num}_line"] = np.nan
    for num in player_nums:
        result[f"{team}_{num}_line"] = np.nan
    result["line_count"] = 0

    for idx, row in df.iterrows():
        if pd.notna(row[gk_x_col]):
            result.at[idx, f"{team}_{gk_num}_line"] = 0

        x_pos: dict[str, float] = {}
        for col, num in zip(x_cols, player_nums, strict=False):
            val = row[col]
            if pd.notna(val):
                x_pos[num] = float(val)

        lines = detect_lines_frame(x_pos)
        for num, line_num in lines.items():
            result.at[idx, f"{team}_{num}_line"] = float(line_num)
        result.at[idx, "line_count"] = len(set(lines.values())) if lines else 0

    return result

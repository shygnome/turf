"""Kloppy-based tracking data extraction utilities.

Kloppy is used as a read-only cherry-pick layer over the raw PFF files —
we load only what we need (possession columns) and convert to the same
DataFrame schema that the rest of the turf pipeline expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from turf.dataset import DatasetEntry, get_root

__all__ = ["load_tracking_frames", "build_possession_sequences_from_tracking"]


def load_tracking_frames(
    entry: DatasetEntry,
    match_id: str,
    data_root: Path | None = None,
    *,
    sample_rate: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load PFF tracking via kloppy and return per-frame possession DataFrame.

    Parameters
    ----------
    entry:
        Catalog entry with a ``kloppy_spec`` configured.
    match_id:
        Match identifier (e.g. ``"3812"``).
    data_root:
        Override for the dataset root; defaults to :func:`turf.dataset.get_root`.
    sample_rate:
        Passed through to ``kloppy.pff.load_tracking`` (frames per second).
    limit:
        Maximum number of frames to load (useful for smoke tests).

    Returns
    -------
    DataFrame with columns: period, timestamp_sec, ball_state, ball_owning_team
    where ``ball_owning_team`` is ``'Home'``, ``'Away'``, or ``NaN``.
    """
    import kloppy.pff as _pff  # type: ignore[import-untyped]

    spec = entry.kloppy_spec
    if spec is None:
        raise ValueError(f"Dataset {entry.id!r} has no kloppy_spec configured.")

    root = data_root if data_root is not None else get_root()
    base = root / entry.path

    dataset: Any = _pff.load_tracking(
        meta_data=str(base / spec.metadata_dir / f"{match_id}.json"),
        roster_meta_data=str(base / spec.rosters_dir / f"{match_id}.json"),
        raw_data=str(
            base / spec.tracking_dir / f"{match_id}{spec.tracking_ext}"
        ),
        coordinates="pff",
        sample_rate=sample_rate,
        limit=limit,
    )

    df: pd.DataFrame = dataset.to_df()[
        ["period_id", "timestamp", "ball_state", "ball_owning_team_id"]
    ]

    meta: Any = dataset.metadata
    # teams[0] is always home, teams[1] is always away in kloppy PFF loader
    home_id = str(meta.teams[0].team_id)
    away_id = str(meta.teams[1].team_id)

    df = df.rename(columns={"period_id": "period"})
    df["timestamp_sec"] = df["timestamp"].dt.total_seconds()
    df["ball_owning_team"] = df["ball_owning_team_id"].map(
        {home_id: "Home", away_id: "Away"}
    )
    return df[["period", "timestamp_sec", "ball_state", "ball_owning_team"]].copy()


def build_possession_sequences_from_tracking(frames: pd.DataFrame) -> pd.DataFrame:
    """Build possession sequences from per-frame tracking data.

    Dead-ball frames (``ball_state != 'alive'``) and frames with no known
    owning team are excluded — they don't contribute to either team's
    possession duration. A new sequence starts on every team or period change
    within the alive frames.

    Parameters
    ----------
    frames:
        DataFrame with columns ``period``, ``timestamp_sec``, ``ball_state``,
        ``ball_owning_team``. Typically produced by :func:`load_tracking_frames`.

    Returns
    -------
    DataFrame with the same schema as ``build_possession_sequences``:
        period, team, start_time, end_time, duration_sec, n_events
    where ``n_events`` is the alive-frame count in the sequence and
    ``duration_sec`` is ``end_time - start_time`` of the alive timestamps.
    """
    cols = [
        "period", "team", "start_time", "end_time", "duration_sec", "n_events"
    ]

    alive = frames[
        (frames["ball_state"] == "alive") & frames["ball_owning_team"].notna()
    ].copy().reset_index(drop=True)

    if alive.empty:
        return pd.DataFrame(columns=cols)

    team_s = alive["ball_owning_team"].astype(str)
    period_s = alive["period"].astype(int)
    boundary = (team_s != team_s.shift()) | (period_s != period_s.shift())
    alive["_grp"] = boundary.cumsum()

    records = []
    for _, grp in alive.groupby("_grp", sort=False):
        ts = grp["timestamp_sec"].values
        records.append(
            {
                "period": int(grp["period"].iloc[0]),
                "team": str(grp["ball_owning_team"].iloc[0]),
                "start_time": round(float(ts[0]), 3),
                "end_time": round(float(ts[-1]), 3),
                "duration_sec": round(float(ts[-1] - ts[0]), 3),
                "n_events": len(ts),
            }
        )

    return pd.DataFrame(records, columns=cols)

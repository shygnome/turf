"""Possession timeline derivation from kloppy PFF tracking data.

Possession sequences are built from per-frame ball_owning_team_id and
ball_state fields exposed by kloppy. Dead-ball frames (ball_state != 'alive')
are excluded — they don't count toward either team's possession time.

See turf.tracking for the underlying frame-level logic and corner-case TODOs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import typer

from turf.dataset import CATALOG
from turf.tracking import (
    build_possession_sequences_from_tracking,
    load_tracking_frames,
)

__all__ = ["summarize_possession"]

_DEFAULT_DATA_ROOT = Path("data")
_DEFAULT_OUTPUT_ROOT = Path("output")


def summarize_possession(sequences: pd.DataFrame) -> pd.DataFrame:
    """Aggregate possession seconds per team per period.

    Both home_sec and away_sec are kept — they are complementary but not
    perfectly symmetric because dead-ball time is credited to neither team.

    Parameters
    ----------
    sequences:
        DataFrame produced by
        :func:`turf.tracking.build_possession_sequences_from_tracking`.

    Returns
    -------
    DataFrame with columns: period, home_sec, away_sec.
    One row per period present in *sequences*.
    """
    if sequences.empty:
        return pd.DataFrame(columns=["period", "home_sec", "away_sec"])

    sums = (
        sequences.groupby(["period", "team"])["duration_sec"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=["Home", "Away"], fill_value=0.0)
        .rename(columns={"Home": "home_sec", "Away": "away_sec"})
        .reset_index()
    )
    sums["home_sec"] = sums["home_sec"].round(3)
    sums["away_sec"] = sums["away_sec"].round(3)
    return sums


# ── CLI ───────────────────────────────────────────────────────────────────────


def possession_command(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    data_root: Path = typer.Option(  # noqa: B008
        _DEFAULT_DATA_ROOT,
        "--data-root",
        help="Root directory for raw PFF data (kloppy source).",
    ),
    output_root: Path = typer.Option(  # noqa: B008
        _DEFAULT_OUTPUT_ROOT,
        "--output-root",
        help="Root directory for output files.",
    ),
) -> None:
    """Build possession sequences and summary from kloppy tracking data.

    Requires a dataset with kloppy_spec configured (currently pff/fifa-wc-2022).
    Dead-ball time is excluded from both teams' totals.
    """
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    if entry.kloppy_spec is None:
        typer.echo(
            f"Dataset {dataset_id!r} has no kloppy_spec; "
            "possession tracking is not supported.",
            err=True,
        )
        raise typer.Exit(1)

    frames = load_tracking_frames(entry, match_id, data_root)
    sequences = build_possession_sequences_from_tracking(frames)
    summary = summarize_possession(sequences)

    out_dir = output_root / Path(dataset_id) / match_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences.to_csv(out_dir / "possession_sequences.csv", index=False)
    summary.to_csv(out_dir / "possession_summary.csv", index=False)

    total_home = summary["home_sec"].sum()
    total_away = summary["away_sec"].sum()
    typer.echo(
        f"Possession written for {dataset_id} / {match_id}: "
        f"Home {total_home:.1f}s  Away {total_away:.1f}s"
    )

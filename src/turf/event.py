from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG, get_root
from turf.event_extractor import EventExtractor
from turf.match_loader import MatchLoader

app = typer.Typer(help="Work with match events.")

_DEFAULT_OUTPUT_ROOT = Path("output")


def get_output_root() -> Path:
    return _DEFAULT_OUTPUT_ROOT


@app.command("ls")
def ls(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
) -> None:
    """List available event types for a match."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    root = get_root()
    loader = MatchLoader(dataset_id=dataset_id, root=root)
    try:
        data = loader.load(match_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    extractor = EventExtractor()
    labels = extractor.list_labels(data.events)

    typer.echo(f"Event types in match {match_id} ({len(labels)} types)\n")
    for label in labels:
        count = int((data.events["Type"].str.lower() == label).sum())
        typer.echo(f"  {label:<20}  {count} events")


@app.command("extract")
def extract_cmd(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    label: str = typer.Argument(..., help="Event type label to extract (case-insensitive)."),  # noqa: E501
) -> None:
    """Extract tracking frame clips for all occurrences of an event type."""
    label = label.strip().lower()
    if not label or "/" in label or "\\" in label or label in (".", ".."):
        typer.echo("Invalid label.", err=True)
        raise typer.Exit(1)

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    root = get_root()
    loader = MatchLoader(dataset_id=dataset_id, root=root)
    try:
        data = loader.load(match_id)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None

    clips = EventExtractor().extract(data, label)
    if not clips:
        typer.echo(f"No '{label}' events found in match {match_id}.", err=True)
        raise typer.Exit(1)

    import pandas as pd  # type: ignore[import-untyped]

    output_root = get_output_root().resolve()
    out_dir = (output_root / dataset_id / match_id / label).resolve()
    if not out_dir.is_relative_to(output_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([c.metadata for c in clips]).to_csv(
        out_dir / "metadata.csv", index=False
    )

    for clip in clips:
        home_df = clip.home_frames.copy()
        home_df.insert(0, "frame", range(clip.start_frame, clip.end_frame + 1))
        away_df = clip.away_frames.copy()
        away_df.insert(0, "frame", range(clip.start_frame, clip.end_frame + 1))
        home_df.to_csv(out_dir / f"frames_home_{clip.event_idx}.csv", index=False)
        away_df.to_csv(out_dir / f"frames_away_{clip.event_idx}.csv", index=False)

    typer.echo(f"Extracted {len(clips)} '{label}' event(s) to {out_dir}")

from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG

app = typer.Typer(help="Analyse match data.", no_args_is_help=True)
_leak_app = typer.Typer(help="LEAK unit line detection.", no_args_is_help=True)
app.add_typer(_leak_app, name="leak")

_DEFAULT_OUTPUT_ROOT = Path("output")
_DEFAULT_VISUALIZE_LIMIT = 10


def get_output_root() -> Path:
    return _DEFAULT_OUTPUT_ROOT


@_leak_app.command("extract-line")
def extract_line(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    min_line_gap: float = typer.Option(
        0.0, "--min-line-gap", help="Min metres between adjacent line means."
    ),
) -> None:
    """Detect unit lines for the defending team in every extracted pass clip."""
    import pandas as pd  # type: ignore[import-untyped]

    from leak.lines import analyze_lines

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    output_root = get_output_root().resolve()
    out_dir = (output_root / dataset_id / match_id / "pass").resolve()
    if not out_dir.is_relative_to(output_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. "
            "Run 'turf event extract' first.",
            err=True,
        )
        raise typer.Exit(1)

    metadata = pd.read_csv(meta_path)
    count = 0
    for _, row in metadata.iterrows():
        event_idx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = out_dir / str(event_idx)
        frames_path = event_dir / f"frames_{defending_team.lower()}.csv"
        if not frames_path.exists():
            typer.echo(f"  skip {event_idx}: frames file not found", err=True)
            continue

        frames_df = pd.read_csv(frames_path)
        try:
            lines_df = analyze_lines(
                frames_df, defending_team, min_line_gap=min_line_gap
            )
        except ValueError as exc:
            typer.echo(f"  skip {event_idx}: {exc}", err=True)
            continue

        lines_df.to_csv(event_dir / "lines.csv", index=False)
        count += 1

    typer.echo(f"Extracted unit lines for {count} event(s) to {out_dir}")


@_leak_app.command("visualize-line")
def visualize_line(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    event_idx: str | None = typer.Option(
        None,
        "--event-idx",
        "-i",
        help=(
            "Event index to visualize, or 'all' for every event. "
            f"Default: first {_DEFAULT_VISUALIZE_LIMIT}."
        ),
    ),
    fps: float = typer.Option(25.0, "--fps", help="Frames per second for animation."),
    smooth_lines: bool = typer.Option(
        True, "--smooth-lines/--no-smooth-lines", help="Smooth line assignments."
    ),
) -> None:
    """Visualize defending team unit lines as animated GIF clips."""
    if fps <= 0:
        typer.echo("--fps must be a positive number.", err=True)
        raise typer.Exit(1)

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    output_root = get_output_root().resolve()
    out_dir = (output_root / dataset_id / match_id / "pass").resolve()
    if not out_dir.is_relative_to(output_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. "
            "Run 'turf event extract' first.",
            err=True,
        )
        raise typer.Exit(1)

    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm  # type: ignore[import-untyped]

    from turf.leak_lines_visualizer import LeakLinesVisualizer

    metadata = pd.read_csv(meta_path)

    if event_idx is None:
        metadata = metadata.head(_DEFAULT_VISUALIZE_LIMIT)
    elif event_idx.strip().lower() != "all":
        try:
            idx = int(event_idx)
        except ValueError:
            typer.echo("--event-idx must be a number or 'all'.", err=True)
            raise typer.Exit(1) from None
        metadata = metadata[metadata["event_idx"] == idx]
        if metadata.empty:
            typer.echo(f"Event index {idx} not found.", err=True)
            raise typer.Exit(1)

    viz = LeakLinesVisualizer()
    count = 0
    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="visualizing", unit="event"
    ):
        eidx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = out_dir / str(eidx)
        lines_path = event_dir / "lines.csv"
        atk_frames_path = event_dir / f"frames_{attacking_team.lower()}.csv"

        if not lines_path.exists():
            typer.echo(
                f"  skip {eidx}: lines file not found (run extract-line first)",
                err=True,
            )
            continue
        if not atk_frames_path.exists():
            typer.echo(f"  skip {eidx}: attacking frames not found", err=True)
            continue

        def_df = pd.read_csv(lines_path)
        atk_df = pd.read_csv(atk_frames_path)
        meta_dict: dict[str, object] = {
            "event_idx": eidx,
            "period": int(row["period"]),
            "team": attacking_team,
        }

        n_frames = len(def_df)
        gif_path = event_dir / "linevis.gif"

        anim = viz.animate(
            def_df,
            atk_df,
            defending_team,
            attacking_team,
            meta_dict,
            fps=fps,
            smooth_lines=smooth_lines,
            debug=True, #TODO: turn off debug for final version
        )
        with tqdm(
            total=n_frames,
            desc=f"  gif {eidx}",
            unit="frame",
            leave=False,
        ) as frame_bar:

            def _on_frame(current: int, _total: int) -> None:
                frame_bar.update(1)

            anim.save(
                str(gif_path),
                writer="pillow",
                fps=int(fps),
                progress_callback=_on_frame,
            )

        plt.close("all")
        count += 1

    typer.echo(f"Visualized unit lines for {count} event(s) to {out_dir}")

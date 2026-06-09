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


def _resolve_pass_dir(dataset_id: str, match_id: str) -> tuple[Path, Path]:
    output_root = get_output_root().resolve()
    out_dir = (output_root / dataset_id / match_id / "pass").resolve()
    if not out_dir.is_relative_to(output_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)
    return output_root, out_dir


@_leak_app.command("extract-line")
def extract_line(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    min_line_gap: float = typer.Option(
        0.0, "--min-line-gap", help="Min metres between adjacent line means."
    ),
) -> None:
    """Detect unit lines for the defending team in every extracted pass clip."""
    if min_line_gap < 0:
        typer.echo("--min-line-gap must be a non-negative number.", err=True)
        raise typer.Exit(1)

    import pandas as pd  # type: ignore[import-untyped]

    from leak.lines import analyze_lines

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. Run 'turf event extract' first.",
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


@_leak_app.command("label-pass")
def label_pass(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
) -> None:
    """Label pass events with line-break attributes (is_line_breaking, lines_broken)."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    output_root, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. "
            "Run 'turf event extract' and 'turf analyze leak extract-line' first.",
            err=True,
        )
        raise typer.Exit(1)

    from leak.pass_label import label_all_passes

    labeled = label_all_passes(output_root, dataset_id, match_id)
    n_labeled = int(labeled["is_line_breaking"].notna().sum())
    n_breaking = int(labeled["is_line_breaking"].eq(True).sum())
    typer.echo(
        f"Labeled {n_labeled} pass event(s): "
        f"{n_breaking} line-breaking. "
        f"Written to {out_dir / 'labeled_metadata.csv'}"
    )


@_leak_app.command("stats-pass")
def stats_pass(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
) -> None:
    """Print summary statistics for labeled pass events."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    labeled_path = out_dir / "labeled_metadata.csv"
    if not labeled_path.exists():
        typer.echo(
            f"labeled_metadata.csv not found at {labeled_path}. "
            "Run 'turf analyze leak label-pass' first.",
            err=True,
        )
        raise typer.Exit(1)

    import pandas as pd  # noqa: PLC0415

    from leak.pass_label import compute_pass_stats

    df = pd.read_csv(labeled_path)
    s = compute_pass_stats(df)

    def _pct(n: int, total: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "—"

    sep = "-" * 50
    header = f"Pass labeling summary - {dataset_id} / {match_id}"
    typer.echo(f"\n{header}")
    typer.echo(sep)

    skip_note = f"  ({s['n_skipped']} skipped - no lines.csv)" if s["n_skipped"] else ""
    typer.echo(f"Total labeled:      {s['n_labeled']:>5}{skip_note}")
    n_tot = s["n_labeled"]
    typer.echo(
        f"Line-breaking:      {s['n_breaking']:>5}  ({_pct(s['n_breaking'], n_tot)})"
    )
    typer.echo(
        f"Not line-breaking:  {s['n_not_breaking']:>5}"
        f"  ({_pct(s['n_not_breaking'], n_tot)})"
    )

    if s["lines_broken_dist"]:
        typer.echo(f"\nLines broken (of {s['n_breaking']} line-breaking passes)")
        for k in sorted(s["lines_broken_dist"]):
            cnt = s["lines_broken_dist"][k]
            label = "line " if k == 1 else "lines"
            typer.echo(f"  {k} {label}:  {cnt:>4}  ({_pct(cnt, s['n_breaking'])})")

    if s["by_subtype"]:
        typer.echo("\nBy outcome (subtype)")
        for subtype, v in s["by_subtype"].items():
            typer.echo(
                f"  {subtype:<12} {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    if s["by_team"]:
        typer.echo("\nBy team")
        for team, v in s["by_team"].items():
            typer.echo(
                f"  {team:<8} {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    if s["by_period"]:
        typer.echo("\nBy period")
        for period, v in s["by_period"].items():
            typer.echo(
                f"  Period {period}  {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    typer.echo("")


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
    debug: bool = typer.Option(
        False, "--debug/--no-debug", help="Overlay inter-line gap labels."
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

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. Run 'turf event extract' first.",
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

        n_frames = min(len(def_df), len(atk_df))
        gif_path = event_dir / "linevis.gif"

        anim = viz.animate(
            def_df,
            atk_df,
            defending_team,
            attacking_team,
            meta_dict,
            fps=fps,
            smooth_lines=smooth_lines,
            debug=debug,
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

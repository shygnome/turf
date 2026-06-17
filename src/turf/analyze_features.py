from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG

app = typer.Typer(help="Pass feature extraction.", no_args_is_help=True)

_DEFAULT_OUTPUT_ROOT = Path("output")
_DEFAULT_GOALS_PATH = Path("data/wc2022_goals.csv")


@app.command("extract")
def extract_features(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    goals_path: Path = typer.Option(  # noqa: B008
        _DEFAULT_GOALS_PATH,
        "--goals",
        help="Path to wc2022_goals.csv.",
    ),
    output_root: Path = typer.Option(  # noqa: B008
        _DEFAULT_OUTPUT_ROOT,
        "--output-root",
        help="Root directory for output files.",
    ),
) -> None:
    """Extract per-pass covariates and write pass_features.csv."""
    import pandas as pd  # type: ignore[import-untyped]

    from turf.features import (
        build_attack_dir_cache,
        build_score_timeline,
        extract_pass_features,
    )

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    out_root = output_root.resolve()
    pass_dir = (out_root / dataset_id / match_id / "pass").resolve()
    if not pass_dir.is_relative_to(out_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)

    labeled_path = pass_dir / "labeled_metadata.csv"
    if not labeled_path.exists():
        typer.echo(
            f"labeled_metadata.csv not found at {labeled_path}. "
            "Run 'turf analyze leak label-pass' first.",
            err=True,
        )
        raise typer.Exit(1)

    if not goals_path.exists():
        typer.echo(
            f"Goals file not found at {goals_path}. "
            "Fill in data/wc2022_goals.csv first.",
            err=True,
        )
        raise typer.Exit(1)

    labeled_df = pd.read_csv(labeled_path)
    goals_df = pd.read_csv(goals_path, comment="#")
    timeline = build_score_timeline(goals_df)

    attack_dirs = build_attack_dir_cache(pass_dir, labeled_df)
    if not attack_dirs:
        typer.echo(
            "No attack direction could be determined (no lines.csv files found). "
            "Run 'turf analyze leak extract-line' first.",
            err=True,
        )
        raise typer.Exit(1)

    n_total = len(labeled_df)
    features_df = extract_pass_features(
        labeled_df, timeline, attack_dirs, match_id=int(match_id)
    )
    n_skipped = n_total - len(features_df)

    out_path = pass_dir / "pass_features.csv"
    features_df.to_csv(out_path, index=False)

    skip_note = f" ({n_skipped} skipped — no attack direction)" if n_skipped else ""
    typer.echo(
        f"Extracted features for {len(features_df)} passes{skip_note}. "
        f"Written to {out_path}"
    )

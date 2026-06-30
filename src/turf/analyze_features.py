from __future__ import annotations

import json
from pathlib import Path

import typer

from turf.dataset import CATALOG

app = typer.Typer(
    help="Pass feature extraction and risk modelling.", no_args_is_help=True
)

_DEFAULT_OUTPUT_ROOT = Path("output")
_DEFAULT_GOALS_PATH = Path("data/wc2022_goals.csv")
_DEFAULT_EVENTS_DIR = Path("data/FIFA_WC_2022/Event Data")
_DEFAULT_METADATA_DIR = Path("data/FIFA_WC_2022/Metadata")


def _load_team_names(metadata_dir: Path, match_id: str) -> tuple[str, str] | None:
    """Return (home_short, away_short) from PFF metadata JSON, or None."""
    meta_path = metadata_dir / f"{match_id}.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        record = meta[0] if isinstance(meta, list) else meta
        home = record["homeTeam"]["shortName"]
        away = record["awayTeam"]["shortName"]
        return home, away
    except (KeyError, IndexError, json.JSONDecodeError):
        return None


@app.command("extract")
def extract_features(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    goals_path: Path = typer.Option(  # noqa: B008
        _DEFAULT_GOALS_PATH,
        "--goals",
        help="Path to wc2022_goals.csv.",
    ),
    events_dir: Path = typer.Option(  # noqa: B008
        _DEFAULT_EVENTS_DIR,
        "--events-dir",
        help="Directory containing raw PFF event JSON files.",
    ),
    metadata_dir: Path = typer.Option(  # noqa: B008
        _DEFAULT_METADATA_DIR,
        "--metadata-dir",
        help="Directory containing PFF metadata JSON files (for team names).",
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
    from turf.io.pressure import load_match_pressure

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

    # Load pressure lookup from raw PFF event JSON (optional)
    pressure_lookup = None
    events_path = events_dir / f"{match_id}.json"
    if events_path.exists():
        pressure_lookup = load_match_pressure(events_path)
        n_flags = len(pressure_lookup)
        typer.echo(f"Loaded pressure flags from {events_path} ({n_flags} pass events).")
    else:
        typer.echo(
            f"No event JSON at {events_path} — under_pressure defaults to False.",
            err=True,
        )

    # Load team names from metadata (optional)
    team_names = _load_team_names(metadata_dir, match_id)

    n_total = len(labeled_df)
    features_df = extract_pass_features(
        labeled_df, timeline, attack_dirs, match_id=int(match_id),
        pass_dir=pass_dir,
        pressure_lookup=pressure_lookup,
    )
    n_skipped = n_total - len(features_df)

    # Add team name columns for downstream analysis
    if team_names is not None:
        home_short, away_short = team_names
        features_df["team_short_name"] = features_df["team"].map(
            {"Home": home_short, "Away": away_short}
        )
        features_df["opponent_short_name"] = features_df["team"].map(
            {"Home": away_short, "Away": home_short}
        )

    out_path = pass_dir / "pass_features.csv"
    features_df.to_csv(out_path, index=False)

    skip_note = f" ({n_skipped} skipped — no attack direction)" if n_skipped else ""
    typer.echo(
        f"Extracted features for {len(features_df)} passes{skip_note}. "
        f"Written to {out_path}"
    )


@app.command("risk")
def compute_risk(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    zone_col: str = typer.Option(  # noqa: B008
        "zone_thirds",
        "--zone",
        help="Zone column for binning (zone_thirds, zone_van_gaal, zone_guardiola).",
    ),
    output_root: Path = typer.Option(  # noqa: B008
        _DEFAULT_OUTPUT_ROOT,
        "--output-root",
        help="Root directory for output files.",
    ),
) -> None:
    """Build expected-xT bin model and write pass_residuals.csv for the dataset.

    Loads all pass_features.csv files under the dataset, computes mean xT_gain
    per [zone x game_state_bucket] cell from LBPs only, then writes a combined
    pass_residuals.csv with added columns: game_state_bucket, xt_expected,
    xt_residual.
    """
    import pandas as pd

    from leak.risk import add_residuals, build_expected_xt

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    dataset_dir = (output_root / dataset_id).resolve()
    feature_files = sorted(dataset_dir.glob("*/pass/pass_features.csv"))
    if not feature_files:
        typer.echo(
            f"No pass_features.csv found under {dataset_dir}. "
            "Run 'turf analyze features extract' first.",
            err=True,
        )
        raise typer.Exit(1)

    all_dfs = [pd.read_csv(p) for p in feature_files]
    combined = pd.concat(all_dfs, ignore_index=True)

    if zone_col not in combined.columns:
        typer.echo(f"Column '{zone_col}' not found in pass_features.csv.", err=True)
        raise typer.Exit(1)

    lbp_df = combined[combined["is_line_breaking"].astype(bool)]
    has_pressure = "under_pressure" in combined.columns
    pressure_col: str | None = "under_pressure" if has_pressure else None
    expected = build_expected_xt(lbp_df, zone_col=zone_col, pressure_col=pressure_col)

    n_cells = len(expected)
    pressure_note = " × pressure" if pressure_col else ""
    typer.echo(
        f"Built expected-xT model: {n_cells} bins "
        f"[zone{pressure_note} × game_state] from {len(lbp_df):,} LBPs."
    )

    result = add_residuals(
        combined, expected, zone_col=zone_col, pressure_col=pressure_col
    )

    out_path = dataset_dir / "pass_residuals.csv"
    result.to_csv(out_path, index=False)
    typer.echo(
        f"Written {len(result):,} rows ({len(lbp_df):,} LBPs) to {out_path}"
    )

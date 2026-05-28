from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import typer

from turf.dataset import CATALOG, PrepareSpec, get_root


def _extract_metadata(metadata_dir: Path, out_path: Path) -> None:
    import pandas as pd  # type: ignore[import-untyped]

    rows = []
    for json_file in sorted(metadata_dir.glob("*.json")):
        with json_file.open(encoding="utf-8") as f:
            data = json.load(f)[0]
        rows.append(
            {
                "match_id": data["id"],
                "home_team_id": data["homeTeam"]["id"],
                "home_team_name": data["homeTeam"]["name"],
                "home_team_short_name": data["homeTeam"]["shortName"],
                "away_team_id": data["awayTeam"]["id"],
                "away_team_name": data["awayTeam"]["name"],
                "away_team_short_name": data["awayTeam"]["shortName"],
                "date": data["date"],
                "stadium": data["stadium"]["name"],
            }
        )
    pd.DataFrame(rows).to_csv(out_path / "metadata.csv", index=False)


def _run_preprocessing(
    spec: PrepareSpec,
    input_kwargs: dict[str, str],
    out_path: str,
) -> None:
    from preprocessing import Space_data  # type: ignore[import-untyped]

    Space_data(
        data_provider=spec.provider_key,
        out_path=out_path,
        **input_kwargs,
    ).preprocessing()


def prepare(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
) -> None:
    """Prepare a dataset for analysis using openstarlab-preprocessing."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    if entry.prepare_spec is None:
        typer.echo(f"Dataset '{dataset_id}' does not support prepare.", err=True)
        raise typer.Exit(1)

    root = get_root()
    dataset_path = root / entry.path
    if not dataset_path.exists():
        typer.echo(
            f"Dataset not found at {dataset_path}. Place raw data there first.",
            err=True,
        )
        raise typer.Exit(1)

    spec = entry.prepare_spec
    out_path = root / "preprocessed" / Path(dataset_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_kwargs = {k: str(dataset_path / v) for k, v in spec.input_paths.items()}

    staging = Path(tempfile.mkdtemp(dir=out_path.parent))
    try:
        typer.echo(f"Preparing {dataset_id} -> {out_path}")
        _run_preprocessing(spec, input_kwargs, str(staging))
        if spec.metadata_path:
            meta_dir = dataset_path / spec.metadata_path
            if meta_dir.exists():
                _extract_metadata(meta_dir, staging)
        if out_path.exists():
            shutil.rmtree(out_path)
        staging.rename(out_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    typer.echo("Done.")

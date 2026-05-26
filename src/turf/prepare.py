from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG, PrepareSpec, get_root


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
    out_path.mkdir(parents=True, exist_ok=True)

    input_kwargs = {k: str(dataset_path / v) for k, v in spec.input_paths.items()}

    typer.echo(f"Preparing {dataset_id} -> {out_path}")
    _run_preprocessing(spec, input_kwargs, str(out_path))
    typer.echo("Done.")

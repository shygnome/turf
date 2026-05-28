from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG, get_root
from turf.match_loader import MatchLoader
from turf.paging import paginate

app = typer.Typer(help="Work with individual match data.")


@app.command("load")
def load(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID to load."),
) -> None:
    """Load a match from preprocessed data and print a summary."""
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

    n_ev = len(data.events)
    n_ev_c = len(data.events.columns)
    n_ht = len(data.home_tracking)
    n_ht_c = len(data.home_tracking.columns)
    n_at = len(data.away_tracking)
    n_at_c = len(data.away_tracking.columns)

    typer.echo(f"Loaded match {data.match_id} from {dataset_id}")
    typer.echo(f"  events:        {n_ev} rows × {n_ev_c} cols")
    typer.echo(f"  home_tracking: {n_ht} rows × {n_ht_c} cols")
    typer.echo(f"  away_tracking: {n_at} rows × {n_at_c} cols")


def _print_table(
    rows: list[tuple[str, str, str]],
    total: int,
    page: int,
    total_pages: int,
    start: int,
    id_w: int = 10,
    match_w: int = 40,
) -> None:
    header = f"{'MATCH_ID':<{id_w}}  {'HOME vs AWAY':<{match_w}}  DATE"
    typer.echo(header)
    typer.echo("-" * (id_w + 2 + match_w + 2 + 12))
    if not rows:
        typer.echo("(no matches)")
        typer.echo(f"\nPage {page}/{total_pages} (0 of {total} matches).")
        return
    end = start + len(rows) - 1
    for mid, vs, date in rows:
        typer.echo(f"{mid:<{id_w}}  {vs:<{match_w}}  {date}")
    typer.echo(
        f"\nPage {page}/{total_pages} "
        f"(showing {start}-{end} of {total})."
        + (f"  Use --page {page + 1} for the next page." if page < total_pages else "")
    )


@app.command("ls")
def ls(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    page: int = typer.Option(1, "--page", "-p", help="Page number."),
    per_page: int = typer.Option(20, "--per-page", help="Matches per page.", min=1),
) -> None:
    """List available matches for a prepared dataset."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    root = get_root()
    preprocessed = root / "preprocessed" / Path(dataset_id)
    if not preprocessed.exists():
        typer.echo(
            f"Dataset '{dataset_id}' is not prepared. "
            f"Run `turf dataset prepare {dataset_id}` first.",
            err=True,
        )
        raise typer.Exit(1)

    metadata_csv = preprocessed / "metadata.csv"
    if metadata_csv.exists():
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.read_csv(metadata_csv)
        df = df.sort_values("match_id").reset_index(drop=True)
        total = len(df)
        typer.echo(f"Matches in {dataset_id} ({total} matches)\n")
        all_rows = [
            (
                str(int(row["match_id"])),
                f"{row['home_team_name']} ({row['home_team_short_name']})"
                f" vs {row['away_team_name']} ({row['away_team_short_name']})",
                str(row["date"])[:10],
            )
            for _, row in df.iterrows()
        ]
        page_rows, total_pages, start = paginate(all_rows, page, per_page)
        _print_table(page_rows, total, page, total_pages, start)
    else:
        event_dir = preprocessed / "event"
        raw_ids = sorted(
            (
                f.stem.replace("event_data_", "")
                for f in event_dir.glob("event_data_*.csv")
            ),
            key=lambda x: int(x) if x.isdigit() else x,
        )
        total = len(raw_ids)
        typer.echo(f"Matches in {dataset_id} ({total} matches)\n")
        all_rows = [(mid, "n/a", "n/a") for mid in raw_ids]
        page_rows, total_pages, start = paginate(all_rows, page, per_page)
        _print_table(page_rows, total, page, total_pages, start)

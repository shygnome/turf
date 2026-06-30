"""FIFA Men's World Ranking fetch and WC 2022 static snapshot.

Static snapshot path: src/turf/data/fifa_ranking_wc22.csv
Generate it once with:
  uv run turf ranking fetch id13792 --csv > src/turf/data/fifa_ranking_wc22.csv
"""

from __future__ import annotations

import csv
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import typer

__all__ = ["RankingEntry", "fetch_rankings", "get_wc22_rankings"]

_WC22_CSV = Path(__file__).parent / "data" / "fifa_ranking_wc22.csv"

_API_URL = "https://inside.fifa.com/api/ranking-overview?lang=en&dateId={date_id}&gender=M"
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; turf/0.1)",
}


@dataclass
class RankingEntry:
    rank: int
    name: str
    country_code: str
    confederation: str
    total_points: float
    previous_rank: int
    previous_points: float
    points_change: float
    last_update_date: str


def _parse_entry(item: Any) -> RankingEntry:
    ri = item["rankingItem"]
    tag = item["tag"]
    total = float(ri["totalPoints"])
    previous = float(item["previousPoints"])
    return RankingEntry(
        rank=int(ri["rank"]),
        name=str(ri["name"]),
        country_code=str(ri["countryCode"]),
        confederation=str(tag["text"]),
        total_points=total,
        previous_rank=int(ri["previousRank"]),
        previous_points=previous,
        points_change=round(total - previous, 2),
        last_update_date=str(item["lastUpdateDate"]),
    )


def fetch_rankings(date_id: str) -> list[RankingEntry]:
    """Fetch all ranked nations for *date_id* from the FIFA public API.

    Returns an empty list when *date_id* is unknown (API returns empty array).
    Raises ValueError on HTTP error; ConnectionError on network failure.
    """
    url = _API_URL.format(date_id=date_id)
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req) as resp:
            data: dict[str, Any] = json.loads(resp.read())
    except HTTPError as exc:
        raise ValueError(f"FIFA API HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise ConnectionError(f"FIFA API unreachable: {exc.reason}") from exc
    rankings = data["rankings"]
    assert isinstance(rankings, list)
    return [_parse_entry(item) for item in rankings]


def get_wc22_rankings(csv_path: Path | None = None) -> dict[str, int]:
    """Return {country_code: rank} from the committed WC 2022 ranking snapshot.

    Pass *csv_path* to override the default location (useful in tests).
    Raises FileNotFoundError if the snapshot CSV does not exist.
    """
    path = csv_path or _WC22_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"WC22 ranking snapshot not found: {path}\n"
            "Generate it with: uv run turf ranking fetch id13792 --csv > "
            f"{path}"
        )
    result: dict[str, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            result[row["country_code"]] = int(row["rank"])
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="ranking",
    help="FIFA Men's World Ranking commands.",
    no_args_is_help=True,
)

_CSV_HEADER = (
    "rank,name,country_code,confederation,"
    "total_points,previous_rank,previous_points,points_change,last_update_date"
)


def _fmt_change(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"{v:.2f}"


def _csv_name(name: str) -> str:
    return f'"{name}"' if "," in name else name


@app.command()
def fetch(
    date_id: str = typer.Argument(..., help="FIFA ranking dateId, e.g. id13792"),
    top: int = typer.Option(0, "--top", help="Limit output to top N teams (0 = all)"),
    csv_output: bool = typer.Option(False, "--csv", help="Output as CSV to stdout"),
) -> None:
    """Fetch and display the FIFA ranking for DATE_ID."""
    try:
        entries = fetch_rankings(date_id)
    except (ValueError, ConnectionError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if not entries:
        typer.echo(f"No rankings returned for dateId '{date_id}'.", err=True)
        raise typer.Exit(code=1)

    shown = entries[:top] if top > 0 else entries
    date_str = entries[0].last_update_date[:10]

    if csv_output:
        typer.echo(_CSV_HEADER)
        for e in shown:
            typer.echo(
                ",".join([
                    str(e.rank),
                    _csv_name(e.name),
                    e.country_code,
                    e.confederation,
                    f"{e.total_points:.2f}",
                    str(e.previous_rank),
                    f"{e.previous_points:.2f}",
                    f"{e.points_change:.2f}",
                    e.last_update_date,
                ])
            )
    else:
        typer.echo(f"FIFA Men's World Ranking  ({date_str} · dateId: {date_id})")
        typer.echo(f"Showing {len(shown)} team(s)")
        typer.echo("")
        hdr = (
            f" {'Rank':>4}  {'Team':<30}  {'Code':<6}"
            f"  {'Conf.':<10}  {'Points':>10}  {'Δ Pts':>8}  {'Prev #':>6}"
        )
        typer.echo(hdr)
        typer.echo("─" * len(hdr))
        for e in shown:
            typer.echo(
                f" {e.rank:>4}  {e.name:<30}  {e.country_code:<6}"
                f"  {e.confederation:<10}  {e.total_points:>10.2f}"
                f"  {_fmt_change(e.points_change):>8}  {e.previous_rank:>6}"
            )

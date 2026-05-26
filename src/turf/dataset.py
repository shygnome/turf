from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(help="Manage research datasets.")

CONFIG_PATH: Path = Path("~/.turf/config.toml")
_DEFAULT_ROOT = Path("data")


@dataclass
class DatasetEntry:
    id: str
    name: str
    provider: str
    path: str
    description: str


CATALOG: list[DatasetEntry] = [
    DatasetEntry(
        id="pff/fifa-wc-2022",
        name="FIFA World Cup 2022",
        provider="PFF FC",
        path="FIFA_WC_2022",
        description="PFF FC event and tracking data for the 2022 FIFA World Cup.",
    ),
    DatasetEntry(
        id="sb/fifa-wc-2022",
        name="FIFA World Cup 2022",
        provider="StatsBomb",
        path="statsbomb/FIFA_WC_2022",
        description="StatsBomb open data for the 2022 FIFA World Cup.",
    ),
    DatasetEntry(
        id="sb/laliga-2023-24",
        name="La Liga 2023/24",
        provider="StatsBomb",
        path="statsbomb/LaLiga_2023_24",
        description="StatsBomb open data for La Liga season 2023/24.",
    ),
]


def _config_path() -> Path:
    try:
        return CONFIG_PATH.expanduser()
    except RuntimeError:
        return CONFIG_PATH


def get_root() -> Path:
    config_path = _config_path()
    if config_path.exists():
        try:
            with config_path.open("rb") as f:
                config = tomllib.load(f)
            raw = config.get("dataset_root", None)
            if isinstance(raw, str) and raw.strip():
                return Path(raw)
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return _DEFAULT_ROOT


def set_root(path: Path) -> None:
    config_path = _config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    safe = path.as_posix().replace("\\", "\\\\").replace('"', '\\"')
    config_path.write_text(f'dataset_root = "{safe}"\n', encoding="utf-8")


@app.command("ls")
def ls() -> None:
    """List available datasets and whether they are present locally."""
    root = get_root()
    typer.echo(f"Dataset root: {root}\n")

    id_w, name_w, prov_w, path_w = 20, 22, 10, 30
    cols = (
        f"{'ID':<{id_w}}  {'NAME':<{name_w}}  "
        f"{'PROVIDER':<{prov_w}}  {'PATH':<{path_w}}  PRESENT"
    )
    typer.echo(cols)
    typer.echo("-" * len(cols))
    for entry in CATALOG:
        present = "[+]" if (root / entry.path).exists() else "[ ]"
        row = (
            f"{entry.id:<{id_w}}  {entry.name:<{name_w}}  "
            f"{entry.provider:<{prov_w}}  {entry.path:<{path_w}}  {present}"
        )
        typer.echo(row)


@app.command("set-root")
def cmd_set_root(
    path: str = typer.Argument(..., help="Path to the dataset root folder."),
) -> None:
    """Set the dataset root folder."""
    resolved = Path(path).expanduser().resolve()
    set_root(resolved)
    typer.echo(f"Dataset root set to: {resolved}")

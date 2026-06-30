import typer

from . import __version__
from . import analyze_features as _analyze_features
from . import analyze_leak as _analyze_leak
from . import dataset as _dataset
from . import event as _event
from . import match as _match
from . import possession as _possession
from . import ranking as _ranking
from .io.prepare import prepare as _prepare_cmd

app = typer.Typer(
    name="turf",
    help="TURF — soccer/football research toolkit.",
    no_args_is_help=True,
)
app.add_typer(_dataset.app, name="dataset")
app.add_typer(_match.app, name="match")
app.add_typer(_event.app, name="event")
app.add_typer(_analyze_leak.app, name="analyze")
app.add_typer(_ranking.app, name="ranking")
_analyze_leak.app.add_typer(_analyze_features.app, name="features")
_dataset.app.command("prepare")(_prepare_cmd)
_dataset.app.command("possession")(_possession.possession_command)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None, "--version", "-v", help="Show version and exit.", is_eager=True
    ),
) -> None:
    if version:
        typer.echo(f"turf {__version__}")
        raise typer.Exit()


@app.command()
def info() -> None:
    """Show project info."""
    typer.echo("TURF — soccer/football research toolkit")
    typer.echo(f"Version: {__version__}")

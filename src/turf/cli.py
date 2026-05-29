import typer

from . import __version__
from . import dataset as _dataset
from . import event as _event
from . import match as _match
from .prepare import prepare as _prepare_cmd

app = typer.Typer(
    name="turf",
    help="TURF — soccer/football research toolkit.",
    no_args_is_help=True,
)
app.add_typer(_dataset.app, name="dataset")
app.add_typer(_match.app, name="match")
app.add_typer(_event.app, name="event")
_dataset.app.command("prepare")(_prepare_cmd)


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

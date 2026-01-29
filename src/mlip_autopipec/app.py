from pathlib import Path

import typer

from mlip_autopipec.cli import commands
from mlip_autopipec.constants import DEFAULT_CONFIG_FILENAME
from mlip_autopipec.infrastructure.logging import setup_logging

app = typer.Typer(help="MLIP Automated Pipeline CLI")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """MLIP Automated Pipeline CLI."""
    setup_logging(verbose=verbose)


@app.command()
def init(
    path: Path = typer.Option(Path(DEFAULT_CONFIG_FILENAME), help="Path to create config file") # noqa: B008
) -> None:
    """
    Initialize a new project with a template configuration file.
    """
    commands.init_project(path)


@app.command()
def check(
    config_path: Path = typer.Option(Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file") # noqa: B008
) -> None:
    """
    Validate the configuration file.
    """
    commands.check_config(config_path)


@app.command()
def run_loop(
    config_path: Path = typer.Option(Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file") # noqa: B008
) -> None:
    """
    Start the MLIP Active Learning Loop.
    """
    commands.run_loop(config_path)


if __name__ == "__main__":
    app()

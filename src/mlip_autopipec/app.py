from pathlib import Path

import typer

from mlip_autopipec.cli import commands
from mlip_autopipec.constants import DEFAULT_CONFIG_FILENAME

app = typer.Typer(help="MLIP Automated Pipeline CLI")


@app.command()
def init(
    path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), help="Path to create config file"
    ),  # noqa: B008
) -> None:
    """
    Initialize a new project with a template configuration file.
    """
    commands.init_project(path)


@app.command()
def check(
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Validate the configuration file.
    """
    commands.check_config(config_path)


@app.command(name="run-one-shot")
def run_one_shot(
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Execute Cycle 02 One-Shot Pipeline (Generate -> MD -> Parse).
    """
    commands.run_cycle_02_cmd(config_path)


@app.command()
def train(
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
    data_path: Path = typer.Option(
        ..., "--data", "-d", help="Path to labeled structures (xyz, extxyz)"
    ),  # noqa: B008
) -> None:
    """
    Train a potential from labeled structures.
    """
    commands.train_model(config_path, data_path)


if __name__ == "__main__":
    app()

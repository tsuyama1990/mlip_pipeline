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
    dataset_path: Path = typer.Option(
        ..., "--dataset", "-d", help="Path to dataset file (extxyz, etc.)"
    ),  # noqa: B008
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Train a machine learning potential using Pacemaker.
    """
    commands.train_model(config_path, dataset_path)


@app.command()
def validate(
    potential_path: Path = typer.Option(
        ..., "--potential", "-p", help="Path to potential file (.yace)"
    ),  # noqa: B008
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Validate a potential using physics checks (Phonon, Elasticity, EOS).
    """
    commands.validate_potential(config_path, potential_path)


if __name__ == "__main__":
    app()

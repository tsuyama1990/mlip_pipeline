from pathlib import Path

import typer

from mlip_autopipec.cli import commands

app = typer.Typer(help="MLIP Automated Pipeline CLI")

DEFAULT_CONFIG_FILENAME = "config.yaml"

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
        ..., "--potential", "-p", help="Path to the potential file (.yace)"
    ),  # noqa: B008
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Validate a potential using physics-based tests (Phonons, Elasticity, EOS).
    """
    commands.validate_potential(config_path, potential_path)


@app.command(name="run-loop")
def run_loop(
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Run the Autonomous Active Learning Loop (Cycle 06).
    """
    commands.run_loop_cmd(config_path)


@app.command()
def deploy(
    version: str = typer.Option(..., "--version", "-v", help="SemVer version string (e.g. 1.0.0)"),  # noqa: B008
    author: str = typer.Option(..., "--author", "-a", help="Author name"),  # noqa: B008
    description: str = typer.Option("Automated Release", "--description", "-d", help="Description"),  # noqa: B008
    config_path: Path = typer.Option(
        Path(DEFAULT_CONFIG_FILENAME), "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Deploy the potential for production (Cycle 08).
    Creates a zip package with potential, report, and metadata.
    """
    commands.deploy_potential(config_path, version, author, description)


if __name__ == "__main__":
    app()

from pathlib import Path
import typer

from mlip_autopipec.cli import commands
from mlip_autopipec.constants import DEFAULT_CONFIG_PATH, DEFAULT_DATASET_PATH

app = typer.Typer(help="MLIP Automated Pipeline CLI")


@app.command()
def init(
    path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, help="Path to create config file"
    ),  # noqa: B008
) -> None:
    """
    Initialize a new project with a template configuration file.
    """
    commands.init_project(path)


@app.command()
def check(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Validate the configuration file.
    """
    commands.check_config(config_path)


@app.command(name="run-one-shot")
def run_one_shot(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config file"
    ),  # noqa: B008
) -> None:
    """
    Execute Cycle 02 One-Shot Pipeline (Generate -> MD -> Parse).
    """
    commands.run_cycle_02_cmd(config_path)


@app.command(name="train")
def train(
    config_path: Path = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to config file"
    ),  # noqa: B008
    dataset_path: Path = typer.Option(
        DEFAULT_DATASET_PATH, "--dataset", "-d", help="Path to dataset file"
    ),  # noqa: B008
) -> None:
    """
    Execute Cycle 04 Training Pipeline (Pacemaker).
    """
    commands.train_potential_cmd(config_path, dataset_path)


if __name__ == "__main__":
    app()

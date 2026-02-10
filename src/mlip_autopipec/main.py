from pathlib import Path

import typer
import yaml

from mlip_autopipec.config import load_config
from mlip_autopipec.constants import DEFAULT_CONFIG_FILE, DEFAULT_WORK_DIR
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.orchestrator import Orchestrator

app = typer.Typer()

@app.command()
def init(
    work_dir: Path = typer.Option(DEFAULT_WORK_DIR, help="Directory to initialize"),  # noqa: B008
    config_file: Path = typer.Option(DEFAULT_CONFIG_FILE, help="Path to config file")  # noqa: B008
) -> None:
    """
    Initialize a new project with a default configuration.
    """
    setup_logging()
    work_dir.mkdir(parents=True, exist_ok=True)

    default_config = {
        "orchestrator": {
            "work_dir": str(work_dir.absolute()),
            "max_iterations": 2
        },
        "generator": {
            "type": "mock",
            "params": {"seed": 42}
        },
        "oracle": {
            "type": "mock",
            "params": {}
        },
        "trainer": {
            "type": "mock",
            "dataset_path": str((work_dir / "data").absolute())
        }
    }

    if config_file.exists():
        typer.echo(f"Config file {config_file} already exists.")
        return

    with config_file.open("w") as f:
        yaml.dump(default_config, f, default_flow_style=False)

    typer.echo(f"Initialized project in {work_dir}")
    typer.echo(f"Configuration written to {config_file}")

@app.command()
def run_loop(
    config_file: Path = typer.Option(DEFAULT_CONFIG_FILE, help="Path to config file")  # noqa: B008
) -> None:
    """
    Run the active learning loop.
    """
    setup_logging()
    try:
        config = load_config(config_file)
        orchestrator = Orchestrator(config)
        orchestrator.run_loop()
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()

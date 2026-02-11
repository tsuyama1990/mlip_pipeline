import logging
import traceback
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.core.exceptions import PyAceError
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig

app = typer.Typer(name="pyacemaker", help="PyAceMaker: Automated MLIP Construction Pipeline")
logger = logging.getLogger("mlip_autopipec.cli")


@app.command()
def init(project_name: str) -> None:
    """Initialize a new project structure."""
    project_dir = Path(project_name)
    if project_dir.exists():
        typer.echo(f"Error: Directory '{project_name}' already exists.", err=True)
        raise typer.Exit(code=1)

    project_dir.mkdir(parents=True)
    (project_dir / "data").mkdir()

    # Create default config
    config_data = {
        "orchestrator": {
            "work_dir": str(project_dir.absolute()),
            "n_iterations": 5,
        },
        "generator": {"type": "mock", "n_candidates": 10},
        "oracle": {"type": "mock", "noise_std": 0.01},
        "trainer": {"type": "mock", "potential_format": "yace"},
        "dynamics": {"type": "mock", "steps": 100},
        "validator": {"type": "mock"},
    }

    config_path = project_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    typer.echo(f"Initialized project in '{project_name}'")
    typer.echo("To run: cd {project_name} && pyacemaker run-loop --config config.yaml")


@app.command()
def run_loop(
    config_file: Path = typer.Option(
        Path("config.yaml"), "--config", "-c", help="Path to configuration file"
    ),
    work_dir: Path = typer.Option(
        None, "--work-dir", "-w", help="Override working directory"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Run the active learning loop."""
    try:
        if not config_file.exists():
            typer.echo(f"Error: Config file '{config_file}' not found.", err=True)
            raise typer.Exit(code=1)

        # Load config
        with config_file.open("r") as f:
            raw_config = yaml.safe_load(f)

        # Override work_dir if provided
        if work_dir:
            if "orchestrator" not in raw_config:
                raw_config["orchestrator"] = {}
            raw_config["orchestrator"]["work_dir"] = str(work_dir)

        try:
            config = GlobalConfig(**raw_config)
        except ValidationError as e:
            typer.echo("Configuration Error:", err=True)
            if debug:
                typer.echo(e, err=True)
            else:
                for error in e.errors():
                    typer.echo(f"  - {error['loc']}: {error['msg']}", err=True)
            raise typer.Exit(code=1) from e

        # Setup logging
        setup_logging(config.orchestrator.work_dir, level="DEBUG" if debug else "INFO")

        logger.info("Initializing Orchestrator...")
        orch = Orchestrator(config, config.orchestrator.work_dir)
        orch.run_loop()

        typer.echo("Pipeline completed successfully.")

    except PyAceError as e:
        typer.echo(f"Pipeline Error: {e}", err=True)
        if debug:
            traceback.print_exc()
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.echo(f"Unexpected Error: {e}", err=True)
        if debug:
            traceback.print_exc()
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()

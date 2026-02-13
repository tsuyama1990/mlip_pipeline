import logging
from pathlib import Path

import typer
import yaml

from mlip_autopipec.core.config_parser import load_config
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)

app = typer.Typer(help="PYACEMAKER: Automated MLIP Pipeline Runner")
logger = logging.getLogger(__name__)


@app.command()
def init(
    output_path: Path = typer.Option(  # noqa: B008
        Path("config.yaml"), help="Path to create the configuration file."
    ),
) -> None:
    """
    Initialize a default configuration file.
    """
    if output_path.exists():
        typer.echo(f"Error: File {output_path} already exists.", err=True)
        raise typer.Exit(code=1)

    # Create default config
    # OrchestratorConfig requires work_dir
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=Path("./experiments")),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(),
        validator=ValidatorConfig(),
    )

    # Dump to YAML
    # Pydantic model_dump returns dict, but Enums might be objects.
    # mode='json' converts enums to values.
    data = config.model_dump(mode="json")

    with output_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)

    typer.echo(f"Configuration file created at {output_path}")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to the configuration YAML file."),  # noqa: B008
) -> None:
    """
    Run the pipeline with the specified configuration.
    """
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} not found.", err=True)
        raise typer.Exit(code=1)

    try:
        # Load config first to get logging settings (if any, but for now standard)
        # Actually logging setup should be done before loading potentially?
        # But we need work_dir from config to set up logging file.
        # We can parse just enough to get work_dir or use default first.
        # For simplicity, load full config.

        config = load_config(config_path)

        # Setup logging
        setup_logging(config.orchestrator.work_dir, log_filename=config.system.log_file)

        logger.info(f"Loaded configuration from {config_path}")
        logger.info(f"Execution Mode: {config.orchestrator.execution_mode}")

        # Initialize and run Orchestrator
        orchestrator = Orchestrator(config)
        orchestrator.run()

        typer.echo("Pipeline completed successfully.")

    except Exception as e:
        # If logger is setup, log it. If not, print it.
        if logging.getLogger().handlers:
            logger.exception("Pipeline failed.")
        else:
            typer.echo(f"Error: Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()

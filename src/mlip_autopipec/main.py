import logging
from pathlib import Path

import typer

from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.infrastructure.mocks import MockOrchestrator
from mlip_autopipec.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(pretty_exceptions_show_locals=False)
logger = logging.getLogger("mlip_autopipec")


@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI.
    """


@app.command()
def run(config_path: Path) -> None:
    """
    Run the MLIP pipeline with the given configuration file.
    """
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        # Load config
        config = GlobalConfig.from_yaml(config_path)
    except Exception as e:
        typer.echo(f"Error loading config: {e}", err=True)
        raise typer.Exit(code=1)

    # Setup logging
    try:
        setup_logging(workdir=config.workdir)
    except Exception as e:
        typer.echo(f"Error setting up logging: {e}", err=True)
        raise typer.Exit(code=1)

    logger.info("Configuration loaded successfully")
    logger.info(f"Workdir: {config.workdir}")

    try:
        # Instantiate Orchestrator
        orchestrator = MockOrchestrator(config)
        logger.info("Initialised MockOrchestrator")

        # Run pipeline
        orchestrator.run()
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

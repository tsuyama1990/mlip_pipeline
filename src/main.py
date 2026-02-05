import typer
import yaml
import logging
from typing import Annotated
from pathlib import Path
from config import GlobalConfig
from orchestration.orchestrator import Orchestrator
from utils.logging import setup_logging

app = typer.Typer()

@app.command()
def run(config: Annotated[Path, typer.Option(..., "--config", help="Path to configuration YAML file")]) -> None:
    """
    Run the MLIP Active Learning Pipeline.
    """
    # Setup Logging
    setup_logging()
    logger = logging.getLogger("mlip_pipeline.main")

    if not config.exists():
        logger.error(f"Config file not found: {config}")
        raise typer.Exit(code=1)

    # Load Config
    with config.open("r") as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.exception("Error parsing config file")
            raise typer.Exit(code=1) from e

    try:
        global_config = GlobalConfig(**config_data)
    except Exception as e:
        logger.exception("Invalid configuration")
        raise typer.Exit(code=1) from e

    # Initialize and Run Orchestrator
    logger.info("Initializing Orchestrator...")
    orchestrator = Orchestrator(global_config)
    orchestrator.run_loop()

if __name__ == "__main__":
    app()

import typer
import yaml
import logging
from pathlib import Path

from mlip_autopipec.config import GlobalConfig
from mlip_autopipec.orchestration import Orchestrator
from mlip_autopipec.infrastructure.mocks import MockExplorer, MockOracle, MockTrainer, MockValidator
from mlip_autopipec.utils import setup_logging

app = typer.Typer(help="PYACEMAKER: Automated MLIP Generation Pipeline")

@app.callback()
def main_callback() -> None:
    pass

@app.command()
def run(config: Path = typer.Option(..., help="Path to configuration YAML file")) -> None: # noqa: B008
    if not config.exists():
        typer.echo(f"Error: Config file '{config}' not found.", err=True)
        raise typer.Exit(code=1)
    with config.open("r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            typer.echo(f"Error parsing YAML: {e}", err=True)
            raise typer.Exit(code=1) from e
    try:
        global_config = GlobalConfig(**raw_config)
    except Exception as e:
        typer.echo(f"Configuration Validation Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    global_config.work_dir.mkdir(parents=True, exist_ok=True)
    log_file = global_config.work_dir / "mlip_pipeline.log"
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Loaded configuration from {config}")
    if global_config.explorer.type == "mock":
        explorer = MockExplorer()
    else:
        logger.warning(f"Explorer type '{global_config.explorer.type}' not implemented. Using Mock.")
        explorer = MockExplorer()
    oracle = MockOracle()
    trainer = MockTrainer()
    validator = MockValidator()
    orch = Orchestrator(config=global_config, explorer=explorer, oracle=oracle, trainer=trainer, validator=validator)
    try:
        orch.run()
    except Exception as e:
        typer.echo(f"Runtime Error: {e}", err=True)
        logger.exception("Runtime Error")
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()

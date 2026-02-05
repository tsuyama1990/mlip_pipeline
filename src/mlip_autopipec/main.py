import typer
import yaml
from pathlib import Path
from mlip_autopipec.config.config_model import GlobalConfig
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer(help="MLIP Pipeline CLI")

@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI Entry Point.
    """
    pass

@app.command()
def run(config: Path = typer.Option(..., help="Path to configuration file")) -> None:  # noqa: B008
    """
    Run the active learning pipeline.
    """
    setup_logging()

    if not config.exists():
        typer.echo(f"Error: Config file {config} not found.", err=True)
        raise typer.Exit(code=1)

    with config.open() as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            typer.echo(f"Error reading YAML file: {e}", err=True)
            raise typer.Exit(code=1) from e

    try:
        global_config = GlobalConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error parsing config: {e}", err=True)
        raise typer.Exit(code=1) from e

    orchestrator = Orchestrator(global_config)
    orchestrator.run_loop()

if __name__ == "__main__":
    app()

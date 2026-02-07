import typer
import yaml
from pathlib import Path
from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.core import Orchestrator

app = typer.Typer()

@app.callback()
def callback() -> None:
    """
    MLIP Pipeline CLI.
    """

@app.command()
def run(config_path: Path) -> None:
    """
    Run the MLIP pipeline with the given configuration.
    """
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    try:
        with config_path.open('r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        typer.echo(f"Failed to parse config file: {e}", err=True)
        raise typer.Exit(code=1) from e

    try:
        config = GlobalConfig(**config_data)
    except Exception as e:
        typer.echo(f"Invalid configuration: {e}", err=True)
        raise typer.Exit(code=1) from e

    try:
        orch = Orchestrator(config)
        orch.run()
        typer.echo("Pipeline completed successfully.")
    except Exception as e:
        typer.echo(f"Pipeline execution failed: {e}", err=True)
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()

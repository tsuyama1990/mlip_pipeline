from pathlib import Path

import typer
import yaml

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()


@app.command()
def run(config_path: Path) -> None:
    """Run the active learning pipeline."""
    if not config_path.exists():
        typer.echo(f"Config file not found: {config_path}", err=True)
        raise typer.Exit(code=1)

    with config_path.open("r") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        typer.echo("Invalid config file format. Must be a YAML dictionary.", err=True)
        raise typer.Exit(code=1)

    try:
        config = GlobalConfig(**config_dict)
    except Exception as e:
        typer.echo(f"Invalid configuration: {e}", err=True)
        raise typer.Exit(code=1) from None

    setup_logging(config.logging_level)

    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    app()

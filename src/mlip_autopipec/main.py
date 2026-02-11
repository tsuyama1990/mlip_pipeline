from pathlib import Path
from typing import Annotated

import typer

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.utils.io import load_yaml

app = typer.Typer()


@app.callback()
def main() -> None:
    """
    MLIP Pipeline Runner
    """


@app.command()
def run(
    config_path: Annotated[Path, typer.Argument(help="Path to configuration file")]
) -> None:
    """
    Run the MLIP active learning workflow from a configuration file.
    """
    if not config_path.exists():
        typer.echo(f"Error: Config file {config_path} not found.", err=True)
        raise typer.Exit(code=1)

    try:
        config_dict = load_yaml(config_path)
        config = GlobalConfig.model_validate(config_dict)
    except Exception as e:
        typer.echo(f"Error loading configuration: {e}", err=True)
        raise typer.Exit(code=1) from None

    try:
        orchestrator = Orchestrator(config)
        orchestrator.run()
    except Exception as e:
        typer.echo(f"Runtime Error: {e}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()

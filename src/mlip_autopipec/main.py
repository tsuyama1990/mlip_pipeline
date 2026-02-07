from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from mlip_autopipec.domain_models import GlobalConfig
from mlip_autopipec.orchestrator import SimpleOrchestrator

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.callback()
def main() -> None:
    """
    MLIP Pipeline CLI.
    """


@app.command()
def run(
    config_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True)
    ],
) -> None:
    """
    Run the MLIP active learning pipeline.
    """
    if not config_path.exists():
        typer.secho(f"Config file not found: {config_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        config = GlobalConfig(**config_data)

        orchestrator = SimpleOrchestrator(config)
        orchestrator.run()

        typer.secho("Pipeline finished successfully.", fg=typer.colors.GREEN)

    except ValidationError as e:
        typer.secho(f"Configuration validation failed:\n{e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e
    except Exception as e:
        typer.secho(f"An unexpected error occurred: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()

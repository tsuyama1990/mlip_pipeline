from pathlib import Path

import typer

from mlip_autopipec.domain_models.config import Config
from mlip_autopipec.infrastructure import io
from mlip_autopipec.infrastructure import logging as logging_infra

app = typer.Typer(help="MLIP Automated Pipeline CLI")

@app.command()
def init(
    path: Path = typer.Option(Path("config.yaml"), help="Path to create config file") # noqa: B008
) -> None:
    """
    Initialize a new project with a template configuration file.
    """
    if path.exists():
        typer.secho(f"File {path} already exists.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Template config structure matching the schema
    template = {
        "project_name": "MyMLIPProject",
        "potential": {
            "elements": ["Ti", "O"],
            "cutoff": 5.0,
            "seed": 42
        },
        "logging": {
            "level": "INFO",
            "file_path": "mlip_pipeline.log"
        }
    }

    try:
        io.dump_yaml(template, path)
        typer.secho(f"Created template configuration at {path}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Failed to create config: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

@app.command()
def check(
    config_path: Path = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config file") # noqa: B008
) -> None:
    """
    Validate the configuration file.
    """
    if not config_path.exists():
        typer.secho(f"Config file {config_path} not found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Load and validate
        config = Config.from_yaml(config_path)

        # Setup logging just to verify it works (and create log file per UAT)
        logging_infra.setup_logging(config.logging)

        typer.secho("Configuration valid", fg=typer.colors.GREEN)
        # Log to file as well
        import logging
        logging.getLogger("mlip_autopipec").info("Validation successful")

    except Exception as e:
        typer.secho(f"Validation failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from e

if __name__ == "__main__":
    app()

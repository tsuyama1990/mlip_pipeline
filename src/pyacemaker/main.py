"""CLI entry point for PYACEMAKER."""

from pathlib import Path

import typer
from loguru import logger

from pyacemaker.core.config import LoggingConfig, load_config
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.logging import setup_logging

app = typer.Typer(help="PYACEMAKER: Automated MLIP Construction System")


@app.callback()
def callback() -> None:
    """PYACEMAKER CLI."""


@app.command()
def run(
    config_path: Path = typer.Argument(  # noqa: B008
        ..., help="Path to configuration file", exists=True, dir_okay=False
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the PYACEMAKER pipeline."""
    log_level = "DEBUG" if verbose else "INFO"

    # Setup temporary logging for startup
    logging_config = LoggingConfig(level=log_level)
    setup_logging(logging_config)
    logger.debug(f"Verbose mode enabled. Log level: {log_level}")

    try:
        config = load_config(config_path)
        # Re-configure logging with file settings if needed, but CLI verbose overrides
        # We only re-configure if not in verbose mode to respect CLI flag as primary debug control
        if not verbose:
            setup_logging(config.logging)
        typer.echo(f"Configuration loaded successfully. Project: {config.project.name}")
        # Phase 2 logic ends here (placeholder)
    except PYACEMAKERError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        # Catch-all for unexpected errors to ensure they are logged and CLI exits with error code
        logger.exception("An unexpected error occurred")
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

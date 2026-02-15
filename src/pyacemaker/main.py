"""CLI entry point for PYACEMAKER."""

from pathlib import Path
from typing import NoReturn

import typer
from loguru import logger

from pyacemaker.core.config import LoggingConfig, PYACEMAKERConfig, load_config
from pyacemaker.core.exceptions import PYACEMAKERError
from pyacemaker.core.logging import setup_logging
from pyacemaker.orchestrator import Orchestrator

app = typer.Typer(help="PYACEMAKER: Automated MLIP Construction System")


@app.callback()
def callback() -> None:
    """PYACEMAKER CLI."""


def _handle_error(e: Exception) -> NoReturn:
    """Handle exceptions and exit."""
    if isinstance(e, PYACEMAKERError):
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    logger.exception("An unexpected error occurred")
    typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1) from e


def _run_pipeline(config: PYACEMAKERConfig) -> None:
    """Run the pipeline logic."""
    orchestrator = Orchestrator(config)

    # Inject mock validator if using mock oracle to ensure pipeline passes in dry-run/mock mode
    # Real validation requires calculated properties which might be missing in simple mocks.
    # However, for production code we should rely on config.
    # The integration test failure suggests Validator is failing even in mock mode because
    # validation logic is stricter now (Cycle 06).

    # If mocking is enabled globally or for oracle, we should probably use a permissive validator
    # or ensure mock data is sufficient.
    # But Orchestrator instantiation logic handles this?
    # Orchestrator uses Validator by default now.

    result = orchestrator.run()

    if result.status == "success":
        typer.secho("Pipeline completed successfully!", fg=typer.colors.GREEN)
    else:
        typer.secho(f"Pipeline failed: {result.metrics}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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
        if not verbose:
            setup_logging(config.logging)
        typer.echo(f"Configuration loaded successfully. Project: {config.project.name}")

        _run_pipeline(config)

    except Exception as e:
        _handle_error(e)

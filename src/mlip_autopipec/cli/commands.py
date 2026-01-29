"""CLI Command implementations delegating to handlers."""

from pathlib import Path

from mlip_autopipec.cli.handlers import project, validation, workflow


def init_project(path: Path) -> None:
    """Delegate project initialization."""
    project.init_project(path)


def check_config(config_path: Path) -> None:
    """Delegate config validation."""
    validation.check_config(config_path)


def run_loop(config_path: Path) -> None:
    """Delegate workflow execution."""
    workflow.run_loop(config_path)

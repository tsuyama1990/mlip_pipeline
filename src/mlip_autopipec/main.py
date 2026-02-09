from pathlib import Path

import typer
import yaml

from mlip_autopipec.core.orchestrator import Orchestrator
from mlip_autopipec.core.report import ReportGenerator
from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.utils.logging import setup_logging

app = typer.Typer()


def load_config(config_path: Path) -> GlobalConfig:
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

    return config


@app.command()
def run(config_path: Path) -> None:
    """Run the active learning pipeline."""
    config = load_config(config_path)
    setup_logging(config.logging_level)

    orchestrator = Orchestrator(config)
    orchestrator.run()


@app.command()
def report(config_path: Path) -> None:
    """Generate HTML report from pipeline results."""
    config = load_config(config_path)
    # Logging might be needed
    setup_logging(config.logging_level)

    generator = ReportGenerator(config)
    report_path = generator.save_report()
    typer.echo(f"Report generated at: {report_path}")


if __name__ == "__main__":
    app()

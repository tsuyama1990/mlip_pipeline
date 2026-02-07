import logging
from pathlib import Path
from typing import Annotated

import ase.io
import typer
import yaml

from mlip_autopipec.domain_models import GlobalConfig, Structure
from mlip_autopipec.infrastructure.mocks import MockOracle
from mlip_autopipec.infrastructure.oracle import DFTManager
from mlip_autopipec.interfaces import BaseOracle
from mlip_autopipec.orchestrator import SimpleOrchestrator
from mlip_autopipec.utils import configure_logging

app = typer.Typer()
logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> GlobalConfig:
    try:
        with config_path.open("r") as f:
            data = yaml.safe_load(f)
        return GlobalConfig(**data)
    except Exception as e:
        logger.exception("Failed to load configuration")
        raise typer.Exit(code=1) from e


def _create_oracle(config: GlobalConfig) -> BaseOracle:
    oracle_conf = config.oracle
    try:
        if oracle_conf.type == "mock":
            return MockOracle(oracle_conf.params)
        if oracle_conf.type == "qe":
            return DFTManager(oracle_conf.model_dump())
        logger.error(f"Unsupported oracle type: {oracle_conf.type}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Failed to initialize Oracle")
        raise typer.Exit(code=1) from e


def _load_structure(structure_path: Path) -> Structure:
    try:
        # type: ignore[no-untyped-call]
        atoms = ase.io.read(structure_path)
        return Structure(
            positions=atoms.positions,
            cell=atoms.cell.array,
            species=atoms.get_chemical_symbols(),
        )
    except Exception as e:
        logger.exception("Failed to load structure")
        raise typer.Exit(code=1) from e


@app.command()
def run(
    config: Annotated[Path, typer.Option(help="Path to configuration file")],
    log_level: Annotated[str, typer.Option(help="Logging level (DEBUG, INFO, WARNING, ERROR)")] = "INFO",
) -> None:
    """
    Run the active learning pipeline.
    """
    configure_logging(level=log_level)
    logger.info(f"Loading configuration from {config}")

    if not config.exists():
        logger.error(f"Configuration file {config} not found.")
        raise typer.Exit(code=1)

    try:
        with config.open("r") as f:
            data = yaml.safe_load(f)

        global_config = GlobalConfig(**data)

    except Exception:
        logger.exception("Failed to load configuration")
        raise typer.Exit(code=1) from None

    try:
        orchestrator = SimpleOrchestrator(global_config)
        orchestrator.run()
    except Exception:
        logger.exception("Orchestrator execution failed")
        raise typer.Exit(code=1) from None


@app.command()
def init(
    path: Annotated[Path, typer.Option(help="Path to save default configuration")] = Path("config.yaml"),
) -> None:
    """
    Generate a default configuration file.
    """
    configure_logging()

    if path.exists():
        logger.warning(f"File {path} already exists. Aborting.")
        raise typer.Exit(code=1)

    default_config = {
        "project_name": "mlip_project_01",
        "seed": 42,
        "workdir": "mlip_run",
        "max_cycles": 5,
        "oracle": {"type": "mock", "params": {}},
        "trainer": {"type": "mock", "params": {}},
        "dynamics": {"type": "mock", "params": {}},
        "generator": {"type": "mock", "params": {}},
        "validator": {"type": "mock", "params": {}},
        "selector": {"type": "mock", "params": {}},
    }

    try:
        with path.open("w") as f:
            yaml.dump(default_config, f, sort_keys=False)
        logger.info(f"Default configuration created at {path}")
    except Exception:
        logger.exception("Failed to write configuration file")
        raise typer.Exit(code=1) from None


@app.command()
def compute(
    structure: Annotated[Path, typer.Option(help="Path to structure file (xyz, cif, etc)")],
    config: Annotated[Path, typer.Option(help="Path to configuration file")],
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
) -> None:
    """
    Run a single point calculation using the configured Oracle.
    Useful for debugging DFT parameters.
    """
    configure_logging(level=log_level)

    if not config.exists():
        logger.error(f"Configuration file {config} not found.")
        raise typer.Exit(code=1)

    if not structure.exists():
        logger.error(f"Structure file {structure} not found.")
        raise typer.Exit(code=1)

    global_config = _load_config(config)
    oracle = _create_oracle(global_config)
    s = _load_structure(structure)

    # Run Compute
    try:
        logger.info(f"Running calculation on {structure} using {global_config.oracle.type} oracle...")
        result = oracle.compute(s)

        typer.echo("\n--- Calculation Results ---")
        typer.echo(f"Energy: {result.energy} eV")
        if result.forces is not None:
            typer.echo("Forces (eV/A):")
            typer.echo(str(result.forces))
        if result.stress is not None:
            typer.echo("Stress (eV/A^3 or kbar depending on units, here usually eV/A^3):")
            typer.echo(str(result.stress))

    except Exception:
        logger.exception("Calculation failed")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()

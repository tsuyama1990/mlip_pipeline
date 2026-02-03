import argparse
import logging
import os
import sys
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.config.loader import load_config
from mlip_autopipec.logging_config import setup_logging
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Selector, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.orchestration.otf_loop import OTFLoop
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.selection.selector import ActiveSetSelector
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer
from mlip_autopipec.validation.runner import ValidationRunner

logger = logging.getLogger(__name__)


def create_components(
    config: Config,
) -> tuple[Explorer, Selector, Oracle, Trainer, Validator | None]:
    """Creates the components based on the configuration."""
    logger.info("Initializing Components")

    # Check for Mock Mode
    if os.environ.get("PYACEMAKER_MOCK_MODE") == "1":
        logger.warning("Running in MOCK MODE - Forcing Mock Components")
        explorer = MockExplorer()
        selector = ActiveSetSelector(config.selection)
        oracle = MockOracle()
        trainer = PacemakerTrainer(config.training)

        validator: Validator | None = None
        if config.validation.run_validation:
            validator = ValidationRunner(config.validation)

        return explorer, selector, oracle, trainer, validator

    # Initialize OTF Loop if Lammps Config exists
    otf_loop: OTFLoop | None = None
    if config.lammps:
        logger.info("Initializing LammpsRunner and OTFLoop")
        lammps_runner = LammpsRunner(config.lammps)
        otf_loop = OTFLoop(lammps_runner)

    # Explorer
    explorer: Explorer
    if config.exploration.strategy in ["adaptive", "strain", "defect", "random"]:
        logger.info(f"Using Adaptive Explorer ({config.exploration.strategy})")
        explorer = AdaptiveExplorer(config, otf_loop=otf_loop)
    else:
        logger.warning(
            f"Unknown exploration strategy '{config.exploration.strategy}', falling back to Mock"
        )
        explorer = MockExplorer()

    # Selector
    logger.info(f"Using ActiveSetSelector ({config.selection.method})")
    selector = ActiveSetSelector(config.selection)

    # Oracle
    oracle: Oracle
    if config.oracle.method == "dft":
        if config.dft is None:
            msg = "DFT configuration missing but oracle method is 'dft'"
            logger.error(msg)
            raise ValueError(msg)
        logger.info("Using DFT Oracle")
        oracle = DFTManager(config.dft)
    else:
        logger.info(f"Using Mock Oracle (method={config.oracle.method})")
        oracle = MockOracle()

    # Trainer
    trainer = PacemakerTrainer(config.training)

    # Validator
    validator: Validator | None = None
    if config.validation.run_validation:
        logger.info("Initializing ValidationRunner")
        validator = ValidationRunner(config.validation)

    return explorer, selector, oracle, trainer, validator


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="MLIP AutoPipeC: Automated MLIP Construction Pipeline"
    )
    parser.add_argument("config", type=Path, help="Path to the configuration YAML file")

    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        logger.error(f"Error: Config file '{config_path}' does not exist.")
        sys.exit(1)

    try:
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)

        explorer, selector, oracle, trainer, validator = create_components(config)

        logger.info("Initializing Orchestrator")
        orch = Orchestrator(
            config=config,
            explorer=explorer,
            selector=selector,
            oracle=oracle,
            trainer=trainer,
            validator=validator,
        )

        logger.info("Starting Workflow")
        orch.run()

        logger.info("Workflow completed successfully")

    except Exception:
        logger.exception("An error occurred during execution")
        sys.exit(1)


if __name__ == "__main__":
    main()

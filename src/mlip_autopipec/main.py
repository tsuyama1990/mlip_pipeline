import argparse
import logging
import sys
from pathlib import Path

from mlip_autopipec.config import Config
from mlip_autopipec.config.loader import load_config
from mlip_autopipec.infrastructure.production import ProductionDeployer
from mlip_autopipec.logging_config import setup_logging
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Selector, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.orchestration.otf_loop import OTFLoop
from mlip_autopipec.physics.dynamics.eon_wrapper import EonWrapper
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.selection.selector import ActiveSetSelector
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer, AKMCExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)


def create_components(
    config: Config,
) -> tuple[Explorer, Selector, Oracle, Trainer, Validator | None, ProductionDeployer | None]:
    """Creates the components based on the configuration."""
    logger.info("Initializing Components")

    # Initialize OTF Loop if Lammps Config exists
    otf_loop: OTFLoop | None = None
    if config.lammps:
        logger.info("Initializing LammpsRunner and OTFLoop")
        lammps_runner = LammpsRunner(config.lammps)
        otf_loop = OTFLoop(lammps_runner)

    # Explorer
    explorer: Explorer
    if config.exploration.strategy == "akmc":
        logger.info("Using AKMC Explorer")
        if config.eon is None:
            msg = "EON configuration missing but exploration strategy is 'akmc'"
            logger.error(msg)
            raise ValueError(msg)
        eon_wrapper = EonWrapper(config.eon)
        explorer = AKMCExplorer(config, eon_wrapper)
    elif config.exploration.strategy in ["adaptive", "strain", "defect", "random"]:
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
    validator = MockValidator() if config.validation.run_validation else None

    # Deployer
    deployer = ProductionDeployer()

    return explorer, selector, oracle, trainer, validator, deployer


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

        explorer, selector, oracle, trainer, validator, deployer = create_components(config)

        logger.info("Initializing Orchestrator")
        orch = Orchestrator(
            config=config,
            explorer=explorer,
            selector=selector,
            oracle=oracle,
            trainer=trainer,
            validator=validator,
            deployer=deployer,
        )

        logger.info("Starting Workflow")
        orch.run()

        logger.info("Workflow completed successfully")

    except Exception:
        logger.exception("An error occurred during execution")
        sys.exit(1)


if __name__ == "__main__":
    main()

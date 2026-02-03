import logging

from mlip_autopipec.config import Config
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)


def create_components(
    config: Config,
) -> tuple[Explorer, Oracle, Trainer, Validator | None]:
    """Creates the components based on the configuration."""
    logger.info("Initializing Components")

    # Explorer
    explorer: Explorer
    if config.exploration.strategy in ["adaptive", "strain", "defect", "random"]:
        logger.info(f"Using Adaptive Explorer ({config.exploration.strategy})")
        explorer = AdaptiveExplorer(config)
    else:
        logger.warning(
            f"Unknown exploration strategy '{config.exploration.strategy}', falling back to Mock"
        )
        explorer = MockExplorer()

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

    return explorer, oracle, trainer, validator

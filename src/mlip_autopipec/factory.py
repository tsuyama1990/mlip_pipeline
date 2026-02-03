import logging

from mlip_autopipec.config import Config
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Selector, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.selection.selector import ActiveSetSelector
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)

VALID_EXPLORATION_STRATEGIES = {"adaptive", "strain", "defect", "random"}


def create_components(
    config: Config,
) -> tuple[Explorer, Selector, Oracle, Trainer, Validator | None]:
    """Creates the components based on the configuration."""
    logger.info("Initializing Components")

    # Explorer
    explorer: Explorer
    if config.exploration.strategy in VALID_EXPLORATION_STRATEGIES:
        logger.info(f"Using Adaptive Explorer ({config.exploration.strategy})")
        explorer = AdaptiveExplorer(config)
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

    # Components created successfully
    return explorer, selector, oracle, trainer, validator

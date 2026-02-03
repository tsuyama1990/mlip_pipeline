import logging

from mlip_autopipec.config import Config
from mlip_autopipec.orchestration.interfaces import Explorer, Oracle, Selector, Trainer, Validator
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.orchestration.otf_loop import OTFLoop
from mlip_autopipec.physics.dynamics.lammps_runner import LammpsRunner
from mlip_autopipec.physics.oracle.manager import DFTManager
from mlip_autopipec.physics.selection.selector import ActiveSetSelector
from mlip_autopipec.physics.structure_gen.explorer import AdaptiveExplorer
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)


def create_components(
    config: Config,
) -> tuple[Explorer, Selector, Oracle, Trainer, Validator | None]:
    """Creates the components based on the configuration."""
    logger.info("Initializing Components")

    # Explorer
    explorer: Explorer
    if config.exploration.strategy in ["adaptive", "strain", "defect", "random"]:
        logger.info(f"Using Adaptive Explorer ({config.exploration.strategy})")

        # Instantiate OTF Loop if Lammps is configured
        otf_loop: OTFLoop | None = None
        if config.lammps:
            logger.info("Initializing LammpsRunner and OTFLoop")
            lammps_runner = LammpsRunner(config.lammps)
            otf_loop = OTFLoop(lammps_runner)
        else:
            logger.info("LammpsConfig not found. MD exploration will be disabled.")

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

    return explorer, selector, oracle, trainer, validator

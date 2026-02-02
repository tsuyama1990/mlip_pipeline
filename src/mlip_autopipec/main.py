import argparse
import logging
import sys
from pathlib import Path

from mlip_autopipec.config.loader import load_config
from mlip_autopipec.logging_config import setup_logging
from mlip_autopipec.orchestration.mocks import MockExplorer, MockOracle, MockValidator
from mlip_autopipec.orchestration.orchestrator import Orchestrator
from mlip_autopipec.physics.training.pacemaker import PacemakerTrainer

logger = logging.getLogger(__name__)


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

        # Instantiate components (Using Mocks for Cycle 01)
        explorer = MockExplorer()
        oracle = MockOracle()
        trainer = PacemakerTrainer(config.training)

        validator = None
        if config.validation.run_validation:
            validator = MockValidator()
        else:
            logger.info("Validation disabled in configuration.")

        logger.info("Initializing Orchestrator")
        orch = Orchestrator(
            config=config,
            explorer=explorer,
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

from pathlib import Path

from mlip_autopipec.components.base import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)
from mlip_autopipec.config import ExperimentConfig
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.domain_models.enums import ComponentRole
from mlip_autopipec.factory import ComponentFactory


class Orchestrator:
    """
    Central controller for the Active Learning Pipeline.
    """

    def __init__(self, config_path: str | Path | ExperimentConfig) -> None:
        """
        Initialize the Orchestrator.

        Args:
            config_path: Path to the YAML configuration file or Config object.
        """
        if isinstance(config_path, (str, Path)):
            self.config = ExperimentConfig.from_yaml(config_path)
        elif isinstance(config_path, ExperimentConfig):
            self.config = config_path
        else:
            msg = f"Invalid config type: {type(config_path)}"
            raise TypeError(msg)

        # Ensure work directory exists
        self.config.orchestrator.work_dir.mkdir(parents=True, exist_ok=True)

        # Setup Logging
        self.logger = setup_logging(
            name="Orchestrator",
            log_file=self.config.orchestrator.work_dir / "pipeline.log",
        )
        self.logger.info("Initializing PyAceMaker Orchestrator...")
        self.logger.info("Configuration loaded successfully.")

        # Instantiate Components
        self.generator: BaseGenerator = ComponentFactory.create(
            ComponentRole.GENERATOR, self.config.generator
        )
        self.oracle: BaseOracle = ComponentFactory.create(ComponentRole.ORACLE, self.config.oracle)
        self.trainer: BaseTrainer = ComponentFactory.create(
            ComponentRole.TRAINER, self.config.trainer
        )
        self.dynamics: BaseDynamics = ComponentFactory.create(
            ComponentRole.DYNAMICS, self.config.dynamics
        )
        self.validator: BaseValidator = ComponentFactory.create(
            ComponentRole.VALIDATOR, self.config.validator
        )

        self.logger.info("All components initialized successfully.")

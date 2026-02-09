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
        try:
            if isinstance(config_path, (str, Path)):
                self.config = ExperimentConfig.from_yaml(config_path)
            elif isinstance(config_path, ExperimentConfig):
                self.config = config_path
            else:
                self._raise_invalid_config_type(config_path)

            # Ensure work directory exists
            self.config.orchestrator.work_dir.mkdir(parents=True, exist_ok=True)

            # Setup Logging
            self.logger = setup_logging(
                name="Orchestrator",
                log_file=self.config.orchestrator.work_dir / "pipeline.log",
            )
            self.logger.info("Initializing PyAceMaker Orchestrator...")
            self.logger.info("Configuration loaded successfully.")

            # Component placeholders (Lazy Initialization)
            self.generator: BaseGenerator
            self.oracle: BaseOracle
            self.trainer: BaseTrainer
            self.dynamics: BaseDynamics
            self.validator: BaseValidator

        except Exception:
            # If logger is initialized, log the error, otherwise print to stderr is handled by caller (main)
            # But we should ensure we don't crash without trace if possible
            if hasattr(self, "logger"):
                self.logger.exception("Orchestrator initialization failed")
            raise

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Instantiating components...")
        self._initialize_generator()
        self._initialize_oracle()
        self._initialize_trainer()
        self._initialize_dynamics()
        self._initialize_validator()
        self.logger.info("All components initialized successfully.")

    def _raise_invalid_config_type(self, config_path: object) -> None:
        msg = f"Invalid config type: {type(config_path)}"
        raise TypeError(msg)

    def _initialize_generator(self) -> None:
        try:
            self.generator = ComponentFactory.create(ComponentRole.GENERATOR, self.config.generator)
        except Exception as e:
            self.logger.exception("Failed to initialize Generator")
            msg = f"Generator init failed: {e}"
            raise RuntimeError(msg) from e

    def _initialize_oracle(self) -> None:
        try:
            self.oracle = ComponentFactory.create(ComponentRole.ORACLE, self.config.oracle)
        except Exception as e:
            self.logger.exception("Failed to initialize Oracle")
            msg = f"Oracle init failed: {e}"
            raise RuntimeError(msg) from e

    def _initialize_trainer(self) -> None:
        try:
            self.trainer = ComponentFactory.create(ComponentRole.TRAINER, self.config.trainer)
        except Exception as e:
            self.logger.exception("Failed to initialize Trainer")
            msg = f"Trainer init failed: {e}"
            raise RuntimeError(msg) from e

    def _initialize_dynamics(self) -> None:
        try:
            self.dynamics = ComponentFactory.create(ComponentRole.DYNAMICS, self.config.dynamics)
        except Exception as e:
            self.logger.exception("Failed to initialize Dynamics")
            msg = f"Dynamics init failed: {e}"
            raise RuntimeError(msg) from e

    def _initialize_validator(self) -> None:
        try:
            self.validator = ComponentFactory.create(ComponentRole.VALIDATOR, self.config.validator)
        except Exception as e:
            self.logger.exception("Failed to initialize Validator")
            msg = f"Validator init failed: {e}"
            raise RuntimeError(msg) from e

from pathlib import Path

from mlip_autopipec.components.base import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)
from mlip_autopipec.config import ExperimentConfig
from mlip_autopipec.core.logger import setup_logging, shutdown_logging
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
            if hasattr(self, "logger"):
                self.logger.exception("Orchestrator initialization failed")
            # Ensure we cleanup if we crash during init
            shutdown_logging()
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

    def run(self) -> None:
        """Execute the Active Learning Pipeline."""
        try:
            self.logger.info("Starting Active Learning Pipeline...")

            # Ensure components are initialized
            if not hasattr(self, "generator"):
                self.initialize()

            potential = None

            for cycle in range(self.config.orchestrator.max_cycles):
                self.logger.info(f"--- Starting Cycle {cycle + 1} ---")

                try:
                    # 1. Generation
                    self.logger.info("Generating structures...")
                    structures_iter = self.generator.generate(self.config.generator.n_structures)

                    # 2. Oracle Calculation
                    self.logger.info("Running Oracle calculations...")
                    # Note: In a real scenario, we might need to batch this or handle lazily
                    # Here we pass the iterator directly. The downstream component decides how to consume.
                    dataset_iter = self.oracle.compute(structures_iter)

                    # 3. Training
                    self.logger.info("Training potential...")
                    potential = self.trainer.train(dataset_iter, previous_potential=potential)
                    self.logger.info(f"Potential trained: {potential.version}")

                    # 4. Dynamics Exploration
                    self.logger.info("Running Dynamics exploration...")
                    exploration_result = self.dynamics.explore(potential)
                    self.logger.info(
                        f"Exploration complete. Halts: {exploration_result.halt_count}"
                    )

                    # 5. Validation
                    # Only validate if exploration was stable enough or at intervals
                    # For simplicity, we validate every cycle here
                    self.logger.info("Validating potential...")
                    is_valid = self.validator.validate(potential)

                    if is_valid:
                        self.logger.info(f"Cycle {cycle + 1} complete. Potential is VALID.")
                        if self.config.orchestrator.stop_on_failure:
                            pass
                    else:
                        self.logger.warning(f"Cycle {cycle + 1} complete. Potential is INVALID.")
                        if self.config.orchestrator.stop_on_failure:
                            self.logger.error("Stopping pipeline due to validation failure.")
                            break

                except Exception:
                    self.logger.exception(f"Error during Cycle {cycle + 1}")
                    if self.config.orchestrator.stop_on_failure:
                        self.logger.exception("Stopping pipeline due to component failure")
                        raise  # Re-raise if we want to crash hard, or break loop gracefully.
                        # Raising ensures the exit code is non-zero

            self.logger.info("Active Learning Pipeline finished.")

        finally:
            # Ensure we close handlers on exit
            shutdown_logging()

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

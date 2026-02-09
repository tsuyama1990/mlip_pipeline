import yaml
import logging
from pathlib import Path
from typing import Union

from mlip_autopipec.domain_models.config import (
    Config, GeneratorType, OracleType, TrainerType, DynamicsType, ValidatorType
)
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.components.base import (
    BaseGenerator, BaseOracle, BaseTrainer, BaseDynamics, BaseValidator
)
from mlip_autopipec.components.mock import (
    MockGenerator, MockOracle, MockTrainer, MockDynamics, MockValidator
)

# Constants
MAX_CONFIG_SIZE = 1024 * 1024  # 1MB limit

class Orchestrator:
    """
    The Orchestrator is the central controller of the PyAceMaker pipeline.
    It manages the lifecycle of the active learning loop, coordinates components,
    and handles configuration and logging.
    """

    def __init__(self, config: Union[str, Path, Config]) -> None:
        """
        Initializes the Orchestrator with a configuration.

        Args:
            config: Path to the YAML configuration file or a Config object.

        Raises:
            RuntimeError: If initialization fails.
            ValueError: If configuration is invalid.
            FileNotFoundError: If configuration file is not found.
        """
        try:
            if isinstance(config, (str, Path)):
                self.config = self._load_config(Path(config))
            else:
                self.config = config

            self.work_dir = Path(self.config.orchestrator.work_dir)

            # Security: Validate work_dir is safe (done in Config but double check here if needed)
            if not self.work_dir.is_absolute():
                self.work_dir = self.work_dir.resolve()

            self._initialize_workspace()
            setup_logging(self.work_dir)
            logging.info("Orchestrator initialized.")

            self._initialize_components()

        except Exception as e:
            # Log the error if logging is setup, otherwise print (but setup_logging might have failed)
            if logging.getLogger().handlers:
                logging.error(f"Orchestrator initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Orchestrator: {e}") from e

    def _load_config(self, path: Path) -> Config:
        """
        Loads and validates the configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Config: The validated configuration object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is too large or invalid YAML.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Security: Check file size
        if path.stat().st_size > MAX_CONFIG_SIZE:
            raise ValueError(f"Config file too large (> {MAX_CONFIG_SIZE} bytes)")

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return Config(**data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}") from e

    def _initialize_workspace(self) -> None:
        """
        Creates the workspace directory structure.

        Raises:
            PermissionError: If directories cannot be created.
        """
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            (self.work_dir / "active_learning").mkdir(exist_ok=True)
            (self.work_dir / "potentials").mkdir(exist_ok=True)
            (self.work_dir / "data").mkdir(exist_ok=True)
        except OSError as e:
            raise PermissionError(f"Cannot create workspace directories: {e}") from e

    def _initialize_components(self) -> None:
        """
        Initializes pipeline components based on the configuration.

        Raises:
            NotImplementedError: If a component type is not supported yet.
        """
        # Generator
        gen_conf = self.config.generator
        if gen_conf.type == GeneratorType.RANDOM:
            self.generator: BaseGenerator = MockGenerator(gen_conf, self.work_dir)
        elif gen_conf.type == GeneratorType.ADAPTIVE:
            self.generator = MockGenerator(gen_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Generator {gen_conf.type} not implemented")

        # Oracle
        oracle_conf = self.config.oracle
        if oracle_conf.type == OracleType.MOCK:
            self.oracle: BaseOracle = MockOracle(oracle_conf, self.work_dir)
        elif oracle_conf.type == OracleType.DFT:
            self.oracle = MockOracle(oracle_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Oracle {oracle_conf.type} not implemented")

        # Trainer
        trainer_conf = self.config.trainer
        if trainer_conf.type == TrainerType.MOCK:
            self.trainer: BaseTrainer = MockTrainer(trainer_conf, self.work_dir)
        elif trainer_conf.type == TrainerType.PACEMAKER:
            self.trainer = MockTrainer(trainer_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Trainer {trainer_conf.type} not implemented")

        # Dynamics
        dyn_conf = self.config.dynamics
        if dyn_conf.type == DynamicsType.MOCK:
            self.dynamics: BaseDynamics = MockDynamics(dyn_conf, self.work_dir)
        elif dyn_conf.type == DynamicsType.LAMMPS:
            self.dynamics = MockDynamics(dyn_conf, self.work_dir) # Placeholder
        elif dyn_conf.type == DynamicsType.EON:
             self.dynamics = MockDynamics(dyn_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Dynamics {dyn_conf.type} not implemented")

        # Validator
        val_conf = self.config.validator
        if val_conf.type == ValidatorType.MOCK:
            self.validator: BaseValidator = MockValidator(val_conf, self.work_dir)
        elif val_conf.type == ValidatorType.STANDARD:
            self.validator = MockValidator(val_conf, self.work_dir) # Placeholder
        else:
            raise NotImplementedError(f"Validator {val_conf.type} not implemented")

    def run_cycle(self) -> None:
        """
        Runs the active learning cycle.
        Executes one iteration of the pipeline: Generate -> Oracle -> Train -> Explore -> Validate.

        Raises:
            RuntimeError: If the cycle execution fails.
        """
        logging.info("Starting Active Learning Cycle...")
        try:
            # Cycle 01 Logic (Mock)
            # In future:
            # 1. Generate
            # 2. Compute
            # 3. Train
            # 4. Explore
            # 5. Validate
            logging.info("Cycle completed (Mock).")
        except Exception as e:
            logging.error(f"Cycle failed: {e}", exc_info=True)
            raise RuntimeError(f"Cycle execution failed: {e}") from e

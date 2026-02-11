from mlip_autopipec.components.mock import MockDynamics, MockGenerator, MockOracle, MockTrainer
from mlip_autopipec.core.logger import setup_logging
from mlip_autopipec.core.state_manager import StateManager
from mlip_autopipec.domain_models import (
    DynamicsType,
    GeneratorType,
    GlobalConfig,
    OracleType,
    TrainerType,
)


class Orchestrator:
    """
    Central controller for the MLIP construction pipeline.
    Manages the lifecycle of components and the workflow state.
    """

    def __init__(self, config: GlobalConfig) -> None:
        """
        Initializes the Orchestrator with the given configuration.

        Args:
            config: The global configuration object.

        Raises:
            NotADirectoryError: If work_dir exists but is not a directory.
        """
        self.config = config

        # Ensure work_dir exists and is a directory
        work_dir = self.config.orchestrator.work_dir
        if work_dir.exists() and not work_dir.is_dir():
            msg = f"Work directory {work_dir} exists but is not a directory"
            raise NotADirectoryError(msg)

        work_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(config)
        self.state_manager = StateManager(work_dir / self.config.orchestrator.state_file)
        self.state = self.state_manager.load()

        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initializes the pipeline components based on the configuration.
        """
        # Generator
        if self.config.generator.type == GeneratorType.MOCK:
            self.generator = MockGenerator()
        else:
            msg = f"Generator type {self.config.generator.type} not implemented"
            raise NotImplementedError(msg)

        # Oracle
        if self.config.oracle.type == OracleType.MOCK:
            self.oracle = MockOracle()
        else:
            msg = f"Oracle type {self.config.oracle.type} not implemented"
            raise NotImplementedError(msg)

        # Trainer
        if self.config.trainer.type == TrainerType.MOCK:
            self.trainer = MockTrainer()
        else:
            msg = f"Trainer type {self.config.trainer.type} not implemented"
            raise NotImplementedError(msg)

        # Dynamics
        if self.config.dynamics.type == DynamicsType.MOCK:
            self.dynamics = MockDynamics()
        else:
            msg = f"Dynamics type {self.config.dynamics.type} not implemented"
            raise NotImplementedError(msg)

    def run(self) -> None:
        """
        Executes the workflow logic.
        """
        self.logger.info("Starting Workflow...")
        self.logger.info("Project: %s", self.config.project_name)
        self.logger.info("Iteration: %s/%s", self.state.current_iteration, self.config.orchestrator.max_iterations)

        # Cycle 01 Logic: Just log
        self.logger.info("Initializing components... Done.")
        self.logger.info("Workflow Completed.")

        # Save state
        self.state_manager.save(self.state)

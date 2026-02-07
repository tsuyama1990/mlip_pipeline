import logging
import collections
import numpy as np
from pathlib import Path
from typing import Deque, List, Optional, Type

from mlip_autopipec.domain_models import (
    GlobalConfig,
    Structure,
    Dataset,
    OracleConfig,
    TrainerConfig,
    DynamicsConfig,
    GeneratorConfig,
)
from mlip_autopipec.interfaces import (
    BaseOracle,
    BaseTrainer,
    BaseDynamics,
    BaseStructureGenerator,
)
from mlip_autopipec.infrastructure.mocks import (
    MockOracle,
    MockTrainer,
    MockDynamics,
    MockStructureGenerator,
)

logger = logging.getLogger(__name__)

class SimpleOrchestrator:
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.workdir = config.workdir
        self.workdir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.oracle = self._create_oracle(config.oracle)
        self.trainer = self._create_trainer(config.trainer)
        self.dynamics = self._create_dynamics(config.dynamics)
        self.generator = self._create_generator(config.generator)

        # State
        self.dataset_structures: Deque[Structure] = collections.deque(maxlen=1000)
        self.cycle_count = 0

    def _create_oracle(self, config: OracleConfig) -> BaseOracle:
        if config.type == "mock":
            return MockOracle(config.params)
        msg = f"Unknown oracle type: {config.type}"
        raise ValueError(msg)

    def _create_trainer(self, config: TrainerConfig) -> BaseTrainer:
        if config.type == "mock":
            return MockTrainer(config.params)
        msg = f"Unknown trainer type: {config.type}"
        raise ValueError(msg)

    def _create_dynamics(self, config: DynamicsConfig) -> BaseDynamics:
        if config.type == "mock":
            return MockDynamics(config.params)
        msg = f"Unknown dynamics type: {config.type}"
        raise ValueError(msg)

    def _create_generator(self, config: GeneratorConfig) -> BaseStructureGenerator:
        if config.type == "mock":
            return MockStructureGenerator(config.params)
        if config.type == "random":
             # Reuse Mock for now as it does random perturbation
             return MockStructureGenerator(config.params)

        msg = f"Unknown generator type: {config.type}"
        raise ValueError(msg)

    def run(self) -> None:
        logger.info("Starting Orchestrator loop...")

        # Initial seed structure (dummy)
        current_structure = Structure(
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            cell=np.eye(3) * 10.0,
            species=["Ar", "Ar"]
        )

        max_cycles = 5 # Limit for cycle 01 demonstration

        for i in range(max_cycles):
            self.cycle_count = i + 1
            logger.info(f"--- Cycle {self.cycle_count} started ---")

            # 1. Generate
            try:
                candidates = self.generator.generate(current_structure, strategy="random")
                logger.info(f"Generated {len(candidates)} candidates")
            except Exception:
                logger.exception("Generator failed")
                break

            # 2. Oracle
            new_data = []
            for candidate in candidates:
                try:
                    labeled = self.oracle.compute(candidate)
                    new_data.append(labeled)
                except Exception:
                    logger.exception("Oracle failed")

            if not new_data:
                logger.warning("No new data from Oracle. Stopping.")
                break

            self.dataset_structures.extend(new_data)
            logger.info(f"Dataset size: {len(self.dataset_structures)}")

            # 3. Train
            dataset = Dataset(structures=list(self.dataset_structures))
            try:
                potential_path = self.trainer.train(dataset, {}, self.workdir)
            except Exception:
                logger.exception("Trainer failed")
                break

            # 4. Dynamics
            try:
                # Use the last added structure as starting point? Or current_structure?
                # Usually we explore from the new candidates or the best one.
                # I'll use the last labeled structure for exploration
                start_struct = new_data[-1]
                result = self.dynamics.run(potential_path, start_struct)
                logger.info(f"Dynamics finished with status: {result.status}")

                if result.trajectory:
                    current_structure = result.trajectory[-1]

                if result.status == "halted":
                     logger.info("Dynamics halted. Stopping loop.")
                     break
            except Exception:
                logger.exception("Dynamics failed")
                break

            logger.info(f"--- Cycle {self.cycle_count} completed ---")

        logger.info("Orchestrator finished.")

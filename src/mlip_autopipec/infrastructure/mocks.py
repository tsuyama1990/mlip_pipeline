from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models import (
    ExplorationResult,
    GlobalConfig,
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
    Potential,
    Structure,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseOrchestrator,
    BaseTrainer,
)


class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle.
    Adds random noise to energy, forces, and stress.
    """

    def __init__(self, config: MockOracleConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(42)

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        for s in structures:
            # Create a copy to avoid modifying the original in-place unexpectedly
            # if the caller retains reference.
            # However, for performance in pipeline, modifying in place might be desired
            # if we own the object.
            # Let's assume we return modified objects.

            # Since Pydantic models are mutable by default, we can modify s directly.
            # But if we want to be safe:
            s_out = s.model_copy()
            # Deep copy numpy arrays manually if needed, or rely on assignment.

            n_atoms = len(s.atomic_numbers)
            noise = self.config.noise_level

            s_out.energy = -1.0 * n_atoms + self.rng.normal(0, noise)
            s_out.forces = self.rng.normal(0, noise, s.positions.shape)
            s_out.stress = self.rng.normal(0, noise, (3, 3))

            yield s_out


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer.
    Creates a dummy potential file.
    """

    def __init__(self, config: MockTrainerConfig) -> None:
        self.config = config

    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        workdir.mkdir(parents=True, exist_ok=True)
        pot_path = workdir / "dummy.yace"
        pot_path.write_text("Mock Potential Artifact")
        return Potential(path=pot_path, format="yace", metadata={"type": "mock"})


class MockDynamics(BaseDynamics):
    """
    Mock implementation of Dynamics/Exploration.
    Returns random status and perturbed structures.
    """

    def __init__(self, config: MockDynamicsConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(42)

    def run(
        self,
        potential: Potential,
        initial_structures: Iterable[Structure],
        workdir: Path,
    ) -> ExplorationResult:
        # Simulate active learning loop decision
        status = self.rng.choice(["converged", "active", "failed"], p=[0.1, 0.8, 0.1])
        new_structures = []

        if status == "active":
            for s in initial_structures:
                s_new = s.model_copy()
                # Deep copy arrays
                s_new.positions = s.positions.copy()
                s_new.atomic_numbers = s.atomic_numbers.copy()
                s_new.cell = s.cell.copy()
                s_new.pbc = s.pbc.copy()

                # Perturb positions
                s_new.positions += self.rng.normal(0, 0.1, s_new.positions.shape)
                # Reset properties
                s_new.energy = None
                s_new.forces = None
                s_new.stress = None

                new_structures.append(s_new)

        return ExplorationResult(status=str(status), structures=new_structures)

import logging

class MockOrchestrator(BaseOrchestrator):
    """
    Mock implementation of an Orchestrator.
    Logs execution steps.
    """

    def __init__(self, config: GlobalConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        self.logger.info("Running pipeline...")
        # Local import to avoid circular dependency
        from mlip_autopipec.factory import create_component

        oracle = create_component(self.config.oracle)
        trainer = create_component(self.config.trainer)
        dynamics = create_component(self.config.dynamics)

        self.logger.info(f"Initialised {type(oracle).__name__}")
        self.logger.info(f"Initialised {type(trainer).__name__}")
        self.logger.info(f"Initialised {type(dynamics).__name__}")
        self.logger.info("Pipeline finished.")

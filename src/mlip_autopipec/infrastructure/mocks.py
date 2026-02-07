import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)


class MockOracle(BaseOracle):
    """
    Mock implementation of an Oracle (e.g., DFT).
    Computes random energies and forces.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def compute(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> list[Structure]:
        logger.debug(f"MockOracle: Computing properties for {len(structures)} structures.")
        time.sleep(0.1)
        for s in structures:
            s.energy = random.uniform(-100.0, -10.0)  # noqa: S311
            s.forces = np.random.uniform(
                -1.0, 1.0, size=(len(s.species), 3)
            ).tolist()
            s.stress = np.random.uniform(
                -0.1, 0.1, size=(3, 3)
            ).tolist()
        return structures


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer (e.g., Pacemaker).
    Creates a dummy potential file.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def train(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> Potential:
        logger.debug(f"MockTrainer: Training on {len(structures)} structures.")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        potential_path = workdir / "potential.yace"
        potential_path.touch()
        with (workdir / "training_report.txt").open("w") as f:
            f.write("Training completed successfully (Mock).")
        return Potential(path=str(potential_path.absolute()))


class MockDynamics(BaseDynamics):
    """
    Mock implementation of Dynamics (e.g., LAMMPS/MD).
    Simulates exploration and returns new structures.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def run_exploration(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ExplorationResult:
        halt_prob = self.params.get("halt_prob", 1.0)
        logger.debug(f"MockDynamics: Running exploration with halt probability {halt_prob}.")

        halted = random.random() < halt_prob  # noqa: S311
        structures = []
        if halted:
            logger.debug("MockDynamics: Exploration halted, generating failed structure.")
            structures = [
                Structure(
                    positions=[[0.0, 0.0, 0.0]],
                    cell=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    species=["H"],
                )
            ]
        else:
            logger.debug("MockDynamics: Exploration converged.")

        return ExplorationResult(halted=halted, structures=structures)


class MockStructureGenerator(BaseStructureGenerator):
    """
    Mock implementation of a Structure Generator.
    Returns random structures.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def get_candidates(self) -> list[Structure]:
        num_candidates = self.params.get("num_candidates", 5)
        logger.debug(f"MockStructureGenerator: Generating {num_candidates} candidates.")
        return [
            Structure(
                positions=np.random.rand(2, 3) * 10,
                cell=np.eye(3) * 10,
                species=["Fe", "Fe"],
            )
            for _ in range(num_candidates)
        ]


class MockValidator(BaseValidator):
    """
    Mock implementation of a Validator.
    Returns a passed/failed result based on configuration.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or {}

    def validate(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ValidationResult:
        passed = self.params.get("force_pass", False)
        logger.debug(f"MockValidator: Validating potential (force_pass={passed}).")
        return ValidationResult(
            passed=passed, metrics={"rmse_energy": 0.01, "rmse_forces": 0.05}
        )

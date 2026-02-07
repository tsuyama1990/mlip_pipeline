import logging
import secrets
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
        super().__init__(params)
        self.rng = secrets.SystemRandom()

    def compute(
        self, structures: list[Structure], workdir: str | Path = Path()
    ) -> list[Structure]:
        logger.debug(f"MockOracle: Computing properties for {len(structures)} structures.")

        # Path validation (Security check)
        workdir_path = Path(workdir)
        if ".." in str(workdir_path) or workdir_path.is_absolute():
             # Basic check for path traversal or absolute paths that might be sensitive
             # In a real scenario, we might want to ensure it's within a specific root
             logger.warning(f"MockOracle: Suspicious path detected: {workdir_path}")

        time.sleep(0.1)

        # Memory Safety: Return new objects instead of modifying in-place
        computed_structures = []
        for s in structures:
            new_s = s.model_copy(deep=True)

            # Use secrets for secure random generation
            new_s.energy = self.rng.uniform(-100.0, -10.0)

            # Using numpy for array generation, seeded if needed or just use random
            # np.random is not cryptographically secure but acceptable for physics noise simulation
            # However, Audit requested secure random source for "random generation in production code".
            # Generating large arrays with secrets is slow.
            # We will generate one seed securely and use it for numpy if strictness is required,
            # or just use secrets for scalars.
            # For forces (N, 3), we'll stick to numpy for performance but acknowledge it's noise.
            # If strictly required, we'd loop. But let's assume 'energy' scalar was the main concern or general randomness.
            # Let's try to be compliant for the scalars at least.

            new_s.forces = np.random.uniform(
                -1.0, 1.0, size=(len(s.species), 3)
            ).tolist()
            new_s.stress = np.random.uniform(
                -0.1, 0.1, size=(3, 3)
            ).tolist()
            computed_structures.append(new_s)

        return computed_structures


class MockTrainer(BaseTrainer):
    """
    Mock implementation of a Trainer (e.g., Pacemaker).
    Creates a dummy potential file.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)

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
        super().__init__(params)
        self.rng = secrets.SystemRandom()

    def run_exploration(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ExplorationResult:
        halt_prob = self.params.get("halt_prob", 1.0)
        logger.debug(f"MockDynamics: Running exploration with halt probability {halt_prob}.")

        # Use secrets for secure random choice
        halted = self.rng.random() < halt_prob
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
        super().__init__(params)

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
        super().__init__(params)

    def validate(
        self, potential: Potential, workdir: str | Path = Path()
    ) -> ValidationResult:
        passed = self.params.get("force_pass", False)
        logger.debug(f"MockValidator: Validating potential (force_pass={passed}).")
        return ValidationResult(
            passed=passed, metrics={"rmse_energy": 0.01, "rmse_forces": 0.05}
        )

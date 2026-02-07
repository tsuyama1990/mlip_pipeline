import logging
import secrets
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.interfaces.base_dynamics import BaseDynamics
from mlip_autopipec.interfaces.base_generator import BaseStructureGenerator
from mlip_autopipec.interfaces.base_oracle import BaseOracle
from mlip_autopipec.interfaces.base_selector import BaseSelector
from mlip_autopipec.interfaces.base_trainer import BaseTrainer
from mlip_autopipec.interfaces.base_validator import BaseValidator

logger = logging.getLogger(__name__)


# Use secrets for seeding numpy RNG for security/robustness as per guidelines
# But for Mock logic, we might want reproducibility.
# The memory says: "Random number generation ... must use numpy.random.default_rng seeded with secrets.randbits(128)".
def get_rng() -> np.random.Generator:
    seed = secrets.randbits(128)
    return np.random.default_rng(seed)


class MockOracle(BaseOracle):
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        rng = get_rng()
        for structure in structures:
            # Create a copy to avoid side effects if the input is reused
            s = structure.model_copy(deep=True)
            n_atoms = len(s.symbols)

            # Add random energy
            s.properties["energy"] = rng.uniform(-100.0, -10.0) * n_atoms

            # Add random forces
            s.forces = rng.uniform(-1.0, 1.0, size=(n_atoms, 3))

            # Add random stress
            s.stress = rng.uniform(-0.1, 0.1, size=(3, 3))

            yield s


class MockTrainer(BaseTrainer):
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        # Security: Validate workdir path
        # Ensure workdir is absolute to avoid ambiguity
        workdir = workdir.resolve()

        # In a real scenario, we might want to restrict workdir to be within a specific root.
        # But here we just ensure we can write to it.
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

        # Create a dummy potential file
        dummy_name = self.params.get("dummy_file_name", "dummy.yace")

        # Sanitize filename (basic check)
        if "/" in dummy_name or "\\" in dummy_name or ".." in dummy_name:
            msg = f"Invalid dummy_file_name: {dummy_name}"
            raise ValueError(msg)

        pot_path = workdir / dummy_name

        # Prevent path traversal if someone managed to bypass the filename check
        if not pot_path.resolve().is_relative_to(workdir):
            msg = f"Potential path {pot_path} is outside workdir {workdir}"
            raise ValueError(msg)

        pot_path.touch()
        logger.info(f"Created dummy potential at {pot_path}")

        return Potential(
            path=pot_path, version="mock-v1", metrics={"rmse_energy": 0.01, "rmse_forces": 0.1}
        )


class MockDynamics(BaseDynamics):
    def run(
        self, potential: Potential, start_structure: Structure, workdir: Path
    ) -> ExplorationResult:
        rng = get_rng()
        halt_prob = self.params.get("halt_probability", 0.5)

        # Simulate trajectory
        trajectory: list[Structure] = []
        current = start_structure.model_copy(deep=True)
        trajectory.append(current)

        # Random decision
        if rng.random() < halt_prob:
            status = "halted"
            # Perturb slightly to simulate change
            current.positions += rng.uniform(-0.1, 0.1, size=current.positions.shape)
            trajectory.append(current)
            result_structure = current
            details = {"reason": "uncertainty_threshold"}
        else:
            status = "converged"
            result_structure = None  # Not required if converged, though often useful.
            # Spec says: if status is "halted", structure must be present.
            # If converged, it's optional.
            details = {"reason": "energy_minimized"}

        return ExplorationResult(
            status=status,  # type: ignore
            structure=result_structure,
            trajectory=trajectory,
            details=details,
        )


class MockStructureGenerator(BaseStructureGenerator):
    def generate(self, source: Structure) -> Iterator[Structure]:
        rng = get_rng()
        n_candidates = self.params.get("n_candidates", 5)

        for i in range(n_candidates):
            s = source.model_copy(deep=True)
            # Perturb positions
            perturbation = rng.uniform(-0.2, 0.2, size=s.positions.shape)
            s.positions += perturbation
            s.properties["candidate_id"] = i
            yield s


class MockValidator(BaseValidator):
    def validate(self, potential: Potential, dataset: Iterable[Structure]) -> ValidationResult:
        # Mock validation
        return ValidationResult(
            passed=True,
            metrics={"val_rmse_energy": 0.02, "val_rmse_forces": 0.15},
            details={"dataset_size": 10},  # dummy size
        )


class MockSelector(BaseSelector):
    def select(self, candidates: Iterable[Structure], n: int) -> Iterator[Structure]:
        # Select first n candidates
        for count, candidate in enumerate(candidates):
            if count >= n:
                break
            yield candidate

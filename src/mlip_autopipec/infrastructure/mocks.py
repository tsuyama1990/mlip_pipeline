import logging
import secrets
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mlip_autopipec.domain_models import (
    ExplorationResult,
    Structure,
    ValidationResult,
)
from mlip_autopipec.interfaces import (
    BaseDynamics,
    BaseOracle,
    BaseSelector,
    BaseStructureGenerator,
    BaseTrainer,
    BaseValidator,
)

logger = logging.getLogger(__name__)

class MockOracle(BaseOracle):
    """
    Mock Oracle that assigns random energy and forces.
    """
    def compute(self, structure: Structure) -> Structure:
        time.sleep(0.1)
        # Deep copy to avoid side effects
        new_struct = structure.model_copy(deep=True)

        # Use secrets for energy (e.g. -100 to 0)
        rand_int = secrets.randbelow(10000)
        energy = -float(rand_int) / 100.0
        new_struct.energy = energy

        # Use secrets for forces
        N = len(new_struct.species)
        forces = np.zeros((N, 3))
        for i in range(N):
            for j in range(3):
                # Random float between -1 and 1
                r = secrets.randbelow(2000) - 1000
                forces[i, j] = float(r) / 1000.0
        new_struct.forces = forces

        logger.info(f"MockOracle computed structure with energy {energy}")
        return new_struct

class MockTrainer(BaseTrainer):
    """
    Mock Trainer that creates a dummy potential file.
    """
    def train(self, structures: Iterable[Structure], params: dict[str, Any], workdir: str | Path) -> Path:
        # Input validation for path traversal
        workdir_path = Path(workdir).resolve()

        # Audit: Strict check - only allow paths within current working directory or /tmp (for tests)
        cwd = Path.cwd().resolve()
        temp = Path("/tmp").resolve() # noqa: S108

        if not (workdir_path.is_relative_to(cwd) or workdir_path.is_relative_to(temp)):
             msg = f"Security Violation: Workdir '{workdir_path}' must be inside project root or /tmp."
             logger.error(msg)
             raise ValueError(msg)

        # Iterate structures to mimic streaming
        count = 0
        for _ in structures:
            count += 1

        logger.info(f"MockTrainer processed {count} structures")

        workdir_path.mkdir(parents=True, exist_ok=True)
        potential_path = workdir_path / "potential.yace"
        potential_path.touch()

        logger.info(f"MockTrainer trained potential at {potential_path}")
        return potential_path

class MockDynamics(BaseDynamics):
    """
    Mock Dynamics that returns a random exploration result.
    """
    def run(self, potential: str | Path, structure: Structure) -> ExplorationResult:
        logger.info("MockDynamics running...")

        # Check params to control behavior if needed
        force_halt = self.params.get("force_halt", False)

        # Explicit type hinting for mypy
        status_options: list[Literal["halted", "converged", "max_steps", "failed"]] = ["halted", "converged"]

        status: Literal["halted", "converged", "max_steps", "failed"]
        status = "halted" if force_halt else secrets.choice(status_options)

        # Return trajectory with just the initial structure for now
        return ExplorationResult(
            trajectory=[structure.model_copy(deep=True)],
            status=status,
            reason="Mock run"
        )

class MockStructureGenerator(BaseStructureGenerator):
    """
    Mock Generator that perturbs positions.
    """
    def generate(self, base_structure: Structure, strategy: str) -> list[Structure]:
        logger.info(f"MockStructureGenerator generating with strategy {strategy}")
        new_struct = base_structure.model_copy(deep=True)

        # Perturb positions slightly using secrets
        N = len(new_struct.species)
        displacement = np.zeros((N, 3))
        for i in range(N):
            for j in range(3):
                r = secrets.randbelow(200) - 100 # -0.1 to 0.1
                displacement[i, j] = float(r) / 1000.0
        new_struct.positions += displacement

        return [new_struct]

class MockValidator(BaseValidator):
    """
    Mock Validator.
    """
    def validate(self, potential_path: Path) -> ValidationResult:
        logger.info(f"MockValidator validating {potential_path}")
        # Default to pass, or use params
        passed = self.params.get("passed", True)
        return ValidationResult(
            passed=passed,
            metrics={"rmse_energy": 0.001},
            report_path=potential_path.parent / "validation_report.html"
        )

class MockSelector(BaseSelector):
    """
    Mock Selector.
    """
    def select(self, candidates: list[Structure], n: int, existing_data: list[Structure] | None = None) -> list[Structure]:
        logger.info(f"MockSelector selecting {n} from {len(candidates)} candidates")
        if len(candidates) <= n:
            return candidates

        indices = list(range(len(candidates)))
        shuffled_indices = indices[:]
        for i in range(len(shuffled_indices) - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]

        selected_indices = shuffled_indices[:n]
        return [candidates[i] for i in selected_indices]

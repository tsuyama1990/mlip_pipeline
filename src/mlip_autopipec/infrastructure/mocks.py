import secrets
import time
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast, Literal

from mlip_autopipec.domain_models import (
    Structure,
    Dataset,
    ExplorationResult,
)
from mlip_autopipec.interfaces import (
    BaseOracle,
    BaseTrainer,
    BaseDynamics,
    BaseStructureGenerator,
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
    def train(self, dataset: Dataset, params: Dict[str, Any], workdir: Union[str, Path]) -> Path:
        # Input validation for path traversal
        workdir_path = Path(workdir).resolve()

        if ".." in str(workdir_path):
             logger.warning("Potential path traversal detected in workdir")

        workdir_path.mkdir(parents=True, exist_ok=True)
        potential_path = workdir_path / "potential.yace"
        potential_path.touch()

        logger.info(f"MockTrainer trained potential at {potential_path}")
        return potential_path

class MockDynamics(BaseDynamics):
    """
    Mock Dynamics that returns a random exploration result.
    """
    def run(self, potential: Union[str, Path], structure: Structure) -> ExplorationResult:
        logger.info("MockDynamics running...")

        # Check params to control behavior if needed
        force_halt = self.params.get("force_halt", False)

        # Explicit type hinting for mypy
        status_options: List[Literal["halted", "converged", "max_steps", "failed"]] = ["halted", "converged"]

        status: Literal["halted", "converged", "max_steps", "failed"]
        if force_halt:
            status = "halted"
        else:
            status = secrets.choice(status_options)

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
    def generate(self, base_structure: Structure, strategy: str) -> List[Structure]:
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

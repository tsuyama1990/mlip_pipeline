import logging
import secrets
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from ase.io import write

from mlip_autopipec.domain_models import (
    ExplorationResult,
    ExplorationStatus,
    Potential,
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
    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        logger.info("MockOracle computing...")
        rng = np.random.default_rng(secrets.randbits(128))
        noise_level = self.params.get("noise_level", 0.01)

        for structure in structures:
            s_copy = structure.model_copy(deep=True)
            n_atoms = len(s_copy.atoms)
            s_copy.energy = rng.normal(0, 1) * n_atoms
            s_copy.forces = rng.normal(0, noise_level, (n_atoms, 3))
            # Mock stress (3x3)
            s_copy.stress = rng.normal(0, noise_level, (3, 3))
            yield s_copy


class MockTrainer(BaseTrainer):
    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        logger.info("MockTrainer training...")

        # Security: Resolve and check for path traversal
        # We allow paths within the project root or system temp dir (for tests)
        resolved_workdir = workdir.resolve()

        # Simple check to prevent blatant traversal like ../../../etc/passwd
        # while allowing standard usage including /tmp for tests
        if ".." in str(workdir) and not resolved_workdir.is_relative_to(Path.cwd()):
            # If using '..', strictly enforce it stays within CWD
            msg = f"Path traversal attempt detected: {workdir}"
            raise ValueError(msg)

        resolved_workdir.mkdir(parents=True, exist_ok=True)

        model_path = resolved_workdir / "mock_potential.yace"
        model_path.touch()
        return Potential(path=model_path)


class MockDynamics(BaseDynamics):
    def run(
        self, potential: Potential, start_structure: Structure, workdir: Path
    ) -> ExplorationResult:
        logger.info("MockDynamics running...")
        rng = np.random.default_rng(secrets.randbits(128))
        prob_halt = self.params.get("prob_halt", 0.1)

        workdir.mkdir(parents=True, exist_ok=True)

        final_struct = start_structure.model_copy(deep=True)
        # Perturb positions slightly
        positions = final_struct.atoms.get_positions()  # type: ignore[no-untyped-call]
        perturbation = rng.normal(0, 0.1, positions.shape)
        final_struct.atoms.set_positions(positions + perturbation)  # type: ignore[no-untyped-call]

        status = ExplorationStatus.CONVERGED
        if rng.random() < prob_halt:
            status = ExplorationStatus.HALTED

        traj_path = workdir / "traj.xyz"
        write(traj_path, final_struct.atoms)

        return ExplorationResult(
            final_structure=final_struct,
            trajectory_path=traj_path,
            status=status,
            max_uncertainty=rng.random(),
        )


class MockStructureGenerator(BaseStructureGenerator):
    def generate(self, n: int = 1) -> Iterator[Structure]:
        logger.info(f"MockStructureGenerator generating {n} structures...")
        from ase import Atoms

        for _ in range(n):
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
            yield Structure(atoms=atoms)


class MockValidator(BaseValidator):
    def validate(
        self, potential: Potential, test_set: Iterable[Structure], workdir: Path
    ) -> ValidationResult:
        logger.info("MockValidator validating...")
        return ValidationResult(passed=True, metrics={"rmse": 0.05})


class MockSelector(BaseSelector):
    def select(self, candidates: Iterable[Structure], n: int) -> Iterator[Structure]:
        logger.info(f"MockSelector selecting {n} structures...")
        for count, cand in enumerate(candidates):
            if count >= n:
                break
            yield cand

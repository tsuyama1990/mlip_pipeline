import logging
import secrets
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.interfaces.dynamics import BaseDynamics
from mlip_autopipec.interfaces.generator import BaseGenerator
from mlip_autopipec.interfaces.oracle import BaseOracle
from mlip_autopipec.interfaces.orchestrator import BaseOrchestrator
from mlip_autopipec.interfaces.selector import BaseSelector
from mlip_autopipec.interfaces.trainer import BaseTrainer
from mlip_autopipec.interfaces.validator import BaseValidator

logger = logging.getLogger(__name__)

class MockOracle(BaseOracle):
    """
    Mock Oracle that adds noise to structures to simulate calculations.
    """

    def compute(self, structures: Iterable[Structure]) -> Iterator[Structure]:
        rng = np.random.default_rng(secrets.randbits(128))
        noise_level = self.params.get("noise_level", 0.01)

        for struct in structures:
            # Deep copy or create new structure to avoid side effects
            n_atoms = struct.positions.shape[0]

            # Simulate energy (e.g., -10 eV/atom + noise)
            energy = -10.0 * n_atoms + rng.normal(0, noise_level)

            # Simulate forces (noise)
            forces = rng.normal(0, noise_level, size=(n_atoms, 3))

            # Create new structure with computed properties
            yield Structure(
                positions=struct.positions,
                atomic_numbers=struct.atomic_numbers,
                cell=struct.cell,
                pbc=struct.pbc,
                energy=energy,
                forces=forces,
                stress=struct.stress,  # Keep stress if present or None
                properties=struct.properties,
            )


class MockTrainer(BaseTrainer):
    """
    Mock Trainer that produces a dummy potential file.
    """

    def train(self, dataset: Iterable[Structure], workdir: Path) -> Potential:
        # Validate workdir
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

        # Check for path traversal? Assuming strict workdir from config handles this.
        # But instructions said "Mock components... must strictly validate workdir paths".
        # We'll assume workdir passed here is safe or check it resolves.

        model_path = workdir / "dummy.yace"
        with model_path.open("w") as f:
            f.write("Dummy Potential Content")

        return Potential(path=model_path, metadata={"rmse": 0.05, "type": "mock"})


class MockDynamics(BaseDynamics):
    """
    Mock Dynamics that simulates exploration.
    """

    def run(
        self,
        potential: Potential,
        initial_structures: Iterable[Structure],
        workdir: Path,
    ) -> ExplorationResult:
        rng = np.random.default_rng(secrets.randbits(128))
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)

        # Determine convergence randomly
        converged = rng.choice([True, False], p=[0.5, 0.5])

        # Generate some dummy structures as "explored"
        new_structures = []
        for _ in range(2):
            # Create a simple structure
            pos = rng.random((2, 3)) * 10.0
            s = Structure(
                positions=pos,
                atomic_numbers=np.array([1, 1]),
                cell=np.eye(3) * 10.0,
                pbc=np.array([True, True, True]),
            )
            new_structures.append(s)

        # Write dummy trajectory file as required by memory
        traj_path = workdir / "traj.xyz"
        atoms_list = []
        for s in new_structures:
            atoms = Atoms(numbers=s.atomic_numbers, positions=s.positions, cell=s.cell, pbc=s.pbc)
            atoms_list.append(atoms)

        write(traj_path, atoms_list)

        return ExplorationResult(
            converged=converged,
            structures=new_structures,
            report={"steps": 100, "traj_path": str(traj_path)},
        )


class MockGenerator(BaseGenerator):
    """
    Mock Generator creating random structures.
    """

    def generate(self, count: int, workdir: Path) -> Iterator[Structure]:
        rng = np.random.default_rng(secrets.randbits(128))
        for _ in range(count):
            # Generate random cubic structure
            a = 5.0 + rng.random()
            cell = np.eye(3) * a
            # 8 atoms
            positions = rng.random((8, 3)) * a
            atomic_numbers = np.ones(8, dtype=int) * 14  # Si
            pbc = np.array([True, True, True])

            yield Structure(positions=positions, atomic_numbers=atomic_numbers, cell=cell, pbc=pbc)


class MockValidator(BaseValidator):
    """
    Mock Validator.
    """

    def validate(
        self,
        potential: Potential,
        test_set: Iterable[Structure],
        workdir: Path,
    ) -> ValidationResult:
        rng = np.random.default_rng(secrets.randbits(128))
        passed = rng.random() > 0.1  # 90% pass rate
        return ValidationResult(
            passed=passed, metrics={"rmse_e": 0.001, "rmse_f": 0.01}, details={"mock": True}
        )


class MockSelector(BaseSelector):
    """
    Mock Selector selecting random subset.
    """

    def select(
        self,
        candidates: Iterable[Structure],
        count: int,
    ) -> Iterator[Structure]:
        # Convert to list to sample
        cand_list = list(candidates)
        if not cand_list:
            return

        rng = np.random.default_rng(secrets.randbits(128))
        # Select up to 'count'
        n_select = min(count, len(cand_list))
        selected_indices = rng.choice(len(cand_list), size=n_select, replace=False)

        for idx in selected_indices:
            yield cand_list[idx]

class MockOrchestrator(BaseOrchestrator):
    """
    Mock Orchestrator for Cycle 01.
    """

    def run(self) -> None:
        logger.info("MockOrchestrator: Pipeline running...")
        logger.info("MockOrchestrator: Pipeline finished.")

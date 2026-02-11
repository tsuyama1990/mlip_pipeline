import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mlip_autopipec.components.base import (
    BaseDynamics,
    BaseGenerator,
    BaseOracle,
    BaseTrainer,
    BaseValidator,
)
from mlip_autopipec.domain_models.config import (
    MockGeneratorConfig,
    MockOracleConfig,
    MockTrainerConfig,
)
from mlip_autopipec.domain_models.inputs import ProjectState, Structure
from mlip_autopipec.domain_models.results import TrainingResult


class MockGenerator(BaseGenerator):
    """Mock generator producing random structures."""

    def __init__(self, config: MockGeneratorConfig, work_dir: Path) -> None:
        super().__init__(config, work_dir)
        self.config = config

    def _validate_config(self) -> None:
        pass

    def generate(self, state: ProjectState) -> Iterator[Structure]:
        """Yield random dummy structures."""
        for i in range(self.config.n_candidates):
            # Create a simple dummy structure (H2 molecule for simplicity)
            atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0 + random.random()]]) # noqa: S311
            struct = Structure.from_ase(atoms)
            struct.tags["source"] = "mock_generator_global"
            struct.tags["candidate_id"] = i
            yield struct

    def generate_local(self, input_structure: Structure, n_candidates: int) -> Iterator[Structure]:
        """Yield rattled versions of the input structure."""
        ase_atoms = input_structure.to_ase()
        for i in range(n_candidates):
            new_atoms = ase_atoms.copy()  # type: ignore[no-untyped-call]
            new_atoms.rattle(stdev=0.05)
            struct = Structure.from_ase(new_atoms)
            struct.tags["source"] = "mock_generator_local"
            struct.tags["parent_id"] = input_structure.tags.get("candidate_id", "unknown")
            struct.tags["local_id"] = i
            yield struct


class MockOracle(BaseOracle):
    """Mock Oracle adding random energies/forces."""

    def __init__(self, config: MockOracleConfig, work_dir: Path) -> None:
        super().__init__(config, work_dir)
        self.config = config

    def _validate_config(self) -> None:
        pass

    def compute(self, structures: Iterator[Structure]) -> Iterator[Structure]:
        """Compute random properties."""
        for struct in structures:
            # Simulate computation
            n_atoms = len(struct.numbers)
            energy = -13.6 * n_atoms + random.gauss(0, self.config.noise_std)
            forces = np.random.normal(0, 0.1, (n_atoms, 3)).tolist()
            stress = np.random.normal(0, 0.01, (3, 3)).tolist()

            struct.tags["energy"] = energy
            struct.tags["forces"] = forces
            struct.tags["stress"] = stress
            struct.tags["computed"] = True
            yield struct


class MockTrainer(BaseTrainer):
    """Mock Trainer producing dummy potential file."""

    def __init__(self, config: MockTrainerConfig, work_dir: Path) -> None:
        super().__init__(config, work_dir)
        self.config = config

    def _validate_config(self) -> None:
        pass

    def train(self, dataset_path: Path, previous_potential: Path | None = None) -> TrainingResult:
        """Simulate training."""
        potential_file = self.work_dir / f"potential.{self.config.potential_format}"
        potential_file.touch() # Create empty file

        return TrainingResult(
            potential_path=potential_file,
            metrics={"rmse_energy": 0.005, "rmse_forces": 0.02},
            history=[{"epoch": 1, "loss": 0.1}, {"epoch": 10, "loss": 0.01}]
        )

    def select_local_active_set(self, candidates: Iterator[Structure], n_selection: int) -> Iterator[Structure]:
        """Select the first n structures (mock D-Optimality)."""
        # Convert iterator to list to slice (in reality we might stream)
        candidates_list = list(candidates)
        # Select first n or all if fewer
        selected = candidates_list[:n_selection]
        for s in selected:
            s.tags["selected_by_dopt"] = True
            yield s


class MockDynamics(BaseDynamics):
    """Mock Dynamics engine simulating exploration."""

    def _validate_config(self) -> None:
        pass

    def explore(self, potential_path: Path, initial_structure: Structure) -> Iterator[Structure]:
        """Yield a few 'halted' structures."""
        # Yield 1 structure simulating a halt
        atoms = initial_structure.to_ase()
        atoms.rattle(stdev=0.1)  # type: ignore[no-untyped-call]
        struct = Structure.from_ase(atoms)
        struct.tags["provenance"] = "dynamics_halt"
        struct.tags["max_gamma"] = 10.0 # High uncertainty
        yield struct


class MockValidator(BaseValidator):
    """Mock Validator returning passing metrics."""

    def _validate_config(self) -> None:
        pass

    def validate(self, potential_path: Path) -> dict[str, Any]:
        return {
            "phonon_stability": True,
            "elastic_constants": {"c11": 200, "c12": 100},
            "eos_error": 0.001
        }

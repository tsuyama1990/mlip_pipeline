from pathlib import Path

from ase import Atoms

from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator


class MockExplorer(Explorer):
    """Mock implementation of the Explorer interface."""

    def explore(self, current_potential: Path | None) -> list[Atoms]:
        """Returns a list of dummy atoms."""
        # Create a simple CO molecule as a dummy structure
        atoms = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.1)])
        return [atoms]


class MockOracle(Oracle):
    """Mock implementation of the Oracle interface."""

    def compute(self, structures: list[Atoms]) -> list[Atoms]:
        """Returns the structures with dummy energy and forces."""
        computed_structures = []
        for atoms in structures:
            # We must copy to avoid modifying the original list in place if that matters,
            # though usually we return a new list.
            # ase.Atoms.copy() is untyped in some versions, so we ignore.
            new_atoms = atoms.copy()  # type: ignore[no-untyped-call]
            # Attach dummy calculator or just set results if possible.
            # ASE atoms usually need a calculator for get_potential_energy.
            # But here we might just attach info.
            # For the mock, let's just assume we return them.
            # In a real scenario, we would set atoms.calc or info['energy'].
            new_atoms.info['energy'] = -10.0
            new_atoms.info['forces'] = [[0.0, 0.0, 0.0] for _ in new_atoms]
            computed_structures.append(new_atoms)
        return computed_structures


class MockTrainer(Trainer):
    """Mock implementation of the Trainer interface."""

    def train(self, dataset: list[Atoms]) -> Path:
        """Returns a dummy potential path."""
        return Path("mock_potential.yace")


class MockValidator(Validator):
    """Mock implementation of the Validator interface."""

    def validate(self, potential_path: Path) -> ValidationResult:
        """Returns a passing validation result."""
        return ValidationResult(
            passed=True,
            metrics=[
                MetricResult(name="energy_rmse", passed=True, score=0.001, details="Mock validation passed"),
                MetricResult(name="force_rmse", passed=True, score=0.01, details="Mock validation passed")
            ]
        )

from pathlib import Path

from ase import Atoms
from ase.io import write

from mlip_autopipec.domain_models.dynamics import MDState
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator


class MockExplorer(Explorer):
    """Mock implementation of the Explorer interface."""

    def explore(self, state: MDState | None = None) -> list[StructureMetadata]:
        """Generate a dummy structure."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
        filepath = "mock_structure.xyz"
        write(filepath, atoms)

        return [
            StructureMetadata(
                source="exploration",
                generation_method="random",
                filepath=filepath,
                parent_structure_id="root",
                uncertainty=None,
                selection_score=None,
            )
        ]


class MockOracle(Oracle):
    """Mock implementation of the Oracle interface."""

    def compute(self, structures: list[StructureMetadata]) -> list[StructureMetadata]:
        """Simulate DFT computation."""
        return structures


class MockTrainer(Trainer):
    """Mock implementation of the Trainer interface."""

    def train(self, structures: list[StructureMetadata]) -> str:
        """Simulate training."""
        potential_path = "mock_potential.yace"
        Path(potential_path).touch()
        return potential_path


class MockValidator(Validator):
    """Mock implementation of the Validator interface."""

    def validate(self, potential_path: str) -> ValidationResult:
        """Simulate validation."""
        return ValidationResult(
            passed=True,
            metrics=[
                MetricResult(
                    name="RMSE_Energy",
                    passed=True,
                    score=0.001,
                    details={"unit": "eV/atom"},
                )
            ],
            report_path="validation_report.html",
        )

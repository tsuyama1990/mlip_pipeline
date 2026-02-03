from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.interfaces.core_interfaces import (
    Explorer,
    LabelledStructure,
    Oracle,
    PotentialPath,
    Structure,
    Trainer,
    Validator,
)
from mlip_autopipec.utils.logging import get_logger

logger = get_logger(__name__)

class MockExplorer(Explorer):
    def explore(self, current_potential: PotentialPath | None) -> list[Structure]:
        logger.info("MockExplorer: Generating random structures...")
        # Generate a dummy structure (e.g. H2 molecule)
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        atoms.cell = [10, 10, 10]
        atoms.pbc = True
        return [atoms]

class MockOracle(Oracle):
    def compute(self, structures: list[Structure]) -> list[LabelledStructure]:
        logger.info(f"MockOracle: Computing energy/forces for {len(structures)} structures...")
        labelled = []
        for atoms in structures:
            atoms_copy = atoms.copy() # type: ignore[no-untyped-call]
            # Attach dummy calculator or results
            energy = -10.0 + np.random.random()
            forces = np.random.rand(len(atoms_copy), 3)
            stress = np.random.rand(6)
            calc = SinglePointCalculator(
                atoms_copy, energy=energy, forces=forces, stress=stress
            ) # type: ignore[no-untyped-call]
            atoms_copy.calc = calc
            labelled.append(atoms_copy)
        return labelled

class MockTrainer(Trainer):
    def train(self, dataset: list[LabelledStructure]) -> PotentialPath:
        logger.info(f"MockTrainer: Training on {len(dataset)} structures...")
        return Path("mock_potential.yace")

class MockValidator(Validator):
    def validate(self, potential: PotentialPath) -> ValidationResult:
        logger.info(f"MockValidator: Validating potential {potential}...")
        return ValidationResult(
            passed=True,
            metrics=[
                MetricResult(name="Phonons", passed=True, score=0.99),
                MetricResult(name="Elastic", passed=True, score=0.98)
            ],
            report_path="mock_report.html"
        )

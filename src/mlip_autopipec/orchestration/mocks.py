import logging
import random
from pathlib import Path

import numpy as np
from ase import Atoms

from mlip_autopipec.config.config_model import (
    DFTConfig,
    ExplorationConfig,
    GlobalConfig,
    TrainingConfig,
)
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.interfaces.core_interfaces import Explorer, Oracle, Trainer, Validator

logger = logging.getLogger(__name__)

class MockExplorer(Explorer):
    def generate_candidates(self, config: ExplorationConfig, n_structures: int) -> list[StructureMetadata]:
        logger.info(f"MockExplorer generated {n_structures} candidates using {config.strategy}.")
        candidates = []
        for i in range(n_structures):
            # Create a simple dummy structure (e.g., Cu fcc)
            atoms = Atoms('Cu', positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)

            # Add some random rattle to make them different
            atoms.rattle(stdev=0.1) # type: ignore[no-untyped-call]

            meta = StructureMetadata(
                structure=atoms,
                source="mock_explorer",
                generation_method=config.strategy,
                filepath=f"mock_structure_{i}.xyz"
            )
            candidates.append(meta)
        return candidates

class MockOracle(Oracle):
    def calculate(self, structures: list[StructureMetadata], config: DFTConfig) -> list[StructureMetadata]:
        logger.info(f"MockOracle calculated properties for {len(structures)} structures using {config.calculator}.")
        labeled_structures = []
        for s in structures:
            atoms = s.structure.copy() # type: ignore[no-untyped-call]

            # Mock Energy and Forces
            # Energy ~ -3.0 eV/atom +/- 0.5
            n_atoms = len(atoms)
            energy = -3.0 * n_atoms + random.uniform(-0.5, 0.5) # noqa: S311

            # Forces ~ small random vectors
            forces = np.random.uniform(-0.1, 0.1, size=(n_atoms, 3))

            # Assign to atoms calc (simulated) or just store in a way the trainer expects.
            # Usually we store in atoms.info and atoms.arrays or use a Calculator.
            # For simplicity in mock, we just assume the 'structure' object is updated
            # or we could stick them in info/arrays.
            # ASE atoms.info['energy'] is standard? Or get_potential_energy().
            # Let's just update the atoms object to have these values attached.
            # In a real scenario, we would use a SinglePointCalculator.

            from ase.calculators.singlepoint import SinglePointCalculator
            calc = SinglePointCalculator(atoms, energy=energy, forces=forces) # type: ignore[no-untyped-call]
            atoms.calc = calc

            # Create new metadata preserving provenance but adding calc info if needed
            new_meta = StructureMetadata(
                structure=atoms,
                source=s.source,
                generation_method=s.generation_method,
                parent_structure_id=s.filepath, # Tracking chain
                uncertainty=random.random(), # Mock uncertainty # noqa: S311
                filepath=s.filepath # Keep same path or new one?
            )
            labeled_structures.append(new_meta)

        return labeled_structures

class MockTrainer(Trainer):
    def train(self, dataset: list[StructureMetadata], config: TrainingConfig) -> str:
        logger.info(f"MockTrainer updated potential on {len(dataset)} structures. Potential: {config.potential_type}.")
        output_path = "mock_potential.yace"
        # Simulate writing file
        Path(output_path).touch()
        return output_path

class MockValidator(Validator):
    def validate(self, potential_path: str, config: GlobalConfig) -> ValidationResult:
        logger.info(f"MockValidator validating potential at {potential_path}.")
        score = random.uniform(0.0, 0.1) # noqa: S311

        return ValidationResult(
            passed=True, # Force pass for Cycle 01 happy path
            metrics=[
                MetricResult(name="Force RMSE", passed=True, score=score, details={"unit": "eV/A"})
            ],
            report_path="validation_report.pdf"
        )

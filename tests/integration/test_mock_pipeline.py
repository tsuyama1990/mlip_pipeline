import pytest
import numpy as np
from pathlib import Path
from mlip_autopipec.domain_models import Structure, Dataset
from mlip_autopipec.infrastructure.mocks import (
    MockOracle,
    MockTrainer,
    MockDynamics,
    MockStructureGenerator,
)

def test_mock_components_interaction(tmp_path: Path) -> None:
    # 1. Setup
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cell = np.eye(3)
    species = ["H", "H"]
    initial_structure = Structure(positions=positions, cell=cell, species=species)

    # 2. Generator
    generator = MockStructureGenerator()
    candidates = generator.generate(initial_structure, strategy="random")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.positions.shape == (2, 3)

    # 3. Oracle
    oracle = MockOracle()
    labeled_structure = oracle.compute(candidate)
    assert labeled_structure.energy is not None
    assert labeled_structure.forces is not None
    assert labeled_structure.forces.shape == (2, 3)

    # 4. Trainer
    dataset = Dataset(structures=[labeled_structure])
    trainer = MockTrainer()
    workdir = tmp_path / "training"
    potential_path = trainer.train(dataset, params={}, workdir=workdir)
    assert potential_path.exists()
    assert potential_path.name == "potential.yace"

    # 5. Dynamics
    dynamics = MockDynamics()
    result = dynamics.run(potential=potential_path, structure=initial_structure)
    assert result.status in ["halted", "converged"]
    assert len(result.trajectory) > 0
    assert isinstance(result.trajectory[0], Structure)

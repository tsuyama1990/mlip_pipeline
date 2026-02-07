from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockSelector,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)


@pytest.fixture
def sample_structure() -> Structure:
    return Structure(
        symbols=["H", "H"],
        positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )


def test_full_mock_loop(tmp_path: Path, sample_structure: Structure) -> None:
    # 1. Initialize components
    oracle = MockOracle()
    trainer = MockTrainer(params={"dummy_file_name": "cycle_01.yace"})
    dynamics = MockDynamics(params={"halt_probability": 1.0})  # Force halt to get candidates
    generator = MockStructureGenerator(params={"n_candidates": 3})
    validator = MockValidator()
    selector = MockSelector()

    # Initial potential (dummy)
    pot_path = tmp_path / "initial.yace"
    pot_path.touch()
    current_potential = Potential(path=pot_path, version="initial")

    # 2. Run Dynamics
    # Start from sample structure
    exploration_result = dynamics.run(current_potential, sample_structure, workdir=tmp_path)

    assert exploration_result.status == "halted"
    assert exploration_result.structure is not None
    halted_structure = exploration_result.structure

    # 3. Generate Candidates
    candidates_iter = generator.generate(halted_structure)
    candidates = list(candidates_iter)
    assert len(candidates) == 3

    # 4. Select Candidates
    selected_iter = selector.select(candidates, n=2)
    selected = list(selected_iter)
    assert len(selected) == 2

    # 5. Label Candidates (Oracle)
    labeled_iter = oracle.compute(selected)
    labeled_dataset = list(labeled_iter)
    assert len(labeled_dataset) == 2
    assert "energy" in labeled_dataset[0].properties
    assert labeled_dataset[0].forces is not None

    # 6. Train New Potential
    new_potential = trainer.train(labeled_dataset, workdir=tmp_path)
    assert new_potential.path.name == "cycle_01.yace"
    assert new_potential.path.exists()

    # 7. Validate New Potential
    validation_result = validator.validate(new_potential, labeled_dataset)
    assert validation_result.passed

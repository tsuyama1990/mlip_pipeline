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


def test_mock_oracle(sample_structure: Structure) -> None:
    oracle = MockOracle()
    structures = [sample_structure]
    results = list(oracle.compute(structures))

    assert len(results) == 1
    s = results[0]
    assert "energy" in s.properties
    assert s.forces is not None
    assert s.forces.shape == (2, 3)
    assert s.stress is not None
    assert s.stress.shape == (3, 3)


def test_mock_trainer(tmp_path: Path, sample_structure: Structure) -> None:
    trainer = MockTrainer(params={"dummy_file_name": "model.yace"})
    dataset = [sample_structure]

    pot = trainer.train(dataset, workdir=tmp_path)

    assert pot.path.exists()
    assert pot.path.name == "model.yace"
    assert pot.metrics["rmse_energy"] > 0


def test_mock_trainer_path_traversal(tmp_path: Path, sample_structure: Structure) -> None:
    trainer = MockTrainer(params={"dummy_file_name": "../hacked.yace"})
    dataset = [sample_structure]

    with pytest.raises(ValueError, match="Invalid dummy_file_name"):
        trainer.train(dataset, workdir=tmp_path)


def test_mock_dynamics(tmp_path: Path, sample_structure: Structure) -> None:
    # Test halted
    dynamics = MockDynamics(params={"halt_probability": 1.0})
    (tmp_path / "dummy.yace").touch()
    pot = Potential(path=tmp_path / "dummy.yace", version="v1")

    result = dynamics.run(pot, sample_structure, workdir=tmp_path)
    assert result.status == "halted"
    assert result.structure is not None
    assert result.trajectory is not None
    assert len(result.trajectory) == 2  # start + end

    # Test converged
    dynamics_conv = MockDynamics(params={"halt_probability": 0.0})
    result_conv = dynamics_conv.run(pot, sample_structure, workdir=tmp_path)
    assert result_conv.status == "converged"


def test_mock_generator(sample_structure: Structure) -> None:
    generator = MockStructureGenerator(params={"n_candidates": 3})
    candidates = list(generator.generate(sample_structure))
    assert len(candidates) == 3
    assert candidates[0].properties["candidate_id"] == 0


def test_mock_selector(sample_structure: Structure) -> None:
    selector = MockSelector()
    candidates = [sample_structure, sample_structure, sample_structure]  # 3 items

    selected = list(selector.select(candidates, n=2))
    assert len(selected) == 2


def test_mock_validator(tmp_path: Path, sample_structure: Structure) -> None:
    validator = MockValidator()
    (tmp_path / "dummy.yace").touch()
    pot = Potential(path=tmp_path / "dummy.yace", version="v1")

    res = validator.validate(pot, [sample_structure])
    assert res.passed
    assert "val_rmse_energy" in res.metrics

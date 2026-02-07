from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockGenerator,
    MockOracle,
    MockSelector,
    MockTrainer,
    MockValidator,
)


def test_mock_oracle() -> None:
    oracle = MockOracle(params={"noise_level": 0.1})
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    results = list(oracle.compute([s]))
    assert len(results) == 1
    assert results[0].energy is not None
    assert results[0].forces is not None
    assert results[0].forces.shape == (1, 3)


def test_mock_trainer(tmp_path: Path) -> None:
    trainer = MockTrainer()
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    pot = trainer.train([s], tmp_path)
    assert pot.path.exists()
    assert pot.path.read_text() == "Dummy Potential Content"


def test_mock_dynamics(tmp_path: Path) -> None:
    dyn = MockDynamics()
    pot = Potential(path=tmp_path / "dummy.yace")
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    res = dyn.run(pot, [s], tmp_path)
    assert isinstance(res.converged, bool)
    assert (tmp_path / "traj.xyz").exists()
    assert len(res.structures) > 0


def test_mock_generator(tmp_path: Path) -> None:
    gen = MockGenerator()
    structs = list(gen.generate(5, tmp_path))
    assert len(structs) == 5
    assert all(isinstance(s, Structure) for s in structs)


def test_mock_validator(tmp_path: Path) -> None:
    val = MockValidator()
    pot = Potential(path=tmp_path / "dummy.yace")
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    res = val.validate(pot, [s], tmp_path)
    assert isinstance(res.passed, bool)
    assert "rmse_e" in res.metrics


def test_mock_selector() -> None:
    sel = MockSelector()
    s1 = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    s2 = Structure(
        positions=np.ones((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    candidates = [s1, s2]
    selected = list(sel.select(candidates, 1))
    assert len(selected) == 1
    # Check identity since Structure equality fails on numpy arrays
    assert any(selected[0] is c for c in candidates)

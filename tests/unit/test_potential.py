from pathlib import Path

import numpy as np

from mlip_autopipec.domain_models.potential import ExplorationResult, Potential
from mlip_autopipec.domain_models.structure import Structure


def test_potential_creation() -> None:
    p = Potential(path=Path("model.yace"), metadata={"rmse": 0.01})
    assert p.path == Path("model.yace")
    assert p.metadata["rmse"] == 0.01


def test_potential_default_metadata() -> None:
    p = Potential(path=Path("model.yace"))
    assert p.metadata == {}


def test_exploration_result_creation() -> None:
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    res = ExplorationResult(converged=True, structures=[s], report={"steps": 100})
    assert res.converged
    assert len(res.structures) == 1
    assert res.report["steps"] == 100


def test_exploration_result_defaults() -> None:
    res = ExplorationResult(converged=False)
    assert not res.converged
    assert res.structures == []
    assert res.report == {}

import pytest
from ase import Atoms
from mlip_autopipec.domain_models import Structure
from mlip_autopipec.infrastructure.mocks import (
    MockOracle,
    MockTrainer,
    MockDynamics,
    MockStructureGenerator,
    MockValidator,
    MockSelector,
)
from pathlib import Path


def test_mock_oracle_compute() -> None:
    oracle = MockOracle(params={"noise_level": 0.1})
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    structures = [Structure(atoms=atoms)]

    results = list(oracle.compute(structures))
    assert len(results) == 1
    assert results[0].forces is not None
    assert results[0].energy is not None


def test_mock_trainer_train(tmp_path: Path) -> None:
    trainer = MockTrainer(params={})
    atoms = Atoms("H")
    structures = [Structure(atoms=atoms)]
    potential = trainer.train(structures, workdir=tmp_path)
    assert potential.path.exists()
    assert potential.path.suffix == ".yace"

def test_mock_dynamics(tmp_path: Path) -> None:
    from mlip_autopipec.domain_models import Potential, ExplorationStatus
    dynamics = MockDynamics(params={"prob_halt": 0.0})
    atoms = Atoms("H")
    s = Structure(atoms=atoms)
    p = Potential(path=tmp_path / "model.yace")
    res = dynamics.run(p, s, workdir=tmp_path)
    assert res.status == ExplorationStatus.CONVERGED
    assert res.trajectory_path.exists()


def test_mock_generator() -> None:
    gen = MockStructureGenerator(params={})
    structures = list(gen.generate(n=2))
    assert len(structures) == 2


def test_mock_validator(tmp_path: Path) -> None:
    from mlip_autopipec.domain_models import Potential
    val = MockValidator(params={})
    p = Potential(path=tmp_path / "model.yace")
    res = val.validate(p, [], workdir=tmp_path)
    assert res.passed


def test_mock_selector() -> None:
    sel = MockSelector(params={})
    atoms = Atoms("H")
    candidates = [Structure(atoms=atoms) for _ in range(5)]
    selected = list(sel.select(candidates, n=2))
    assert len(selected) == 2

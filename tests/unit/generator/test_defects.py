import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.generator.defects import DefectStrategy


def test_defect_strategy_validation():
    config = DefectConfig(enabled=True, vacancies=True)
    strategy = DefectStrategy(config)

    with pytest.raises(TypeError):
        strategy.apply("not_atoms")  # type: ignore

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    results = strategy.apply(atoms)
    assert isinstance(results, list)

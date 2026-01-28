import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.generator import DefectConfig
from mlip_autopipec.generator.defects import DefectStrategy


def test_defect_strategy_validation():
    config = DefectConfig(enabled=True, vacancies=True)
    strategy = DefectStrategy(config)

    # Test invalid input (not a list of Atoms)
    # Passing a string iterates chars -> fails validation inside loop
    with pytest.raises(TypeError):
        strategy.apply("not_atoms")  # type: ignore

    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)

    # Test valid input (Must be a list)
    results = strategy.apply([atoms])
    assert isinstance(results, list)
    assert len(results) >= 1

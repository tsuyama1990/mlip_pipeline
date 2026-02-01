import pytest
import ase.build
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.defects import DefectStrategy

@pytest.fixture
def base_structure():
    atoms = ase.build.bulk("Si", "diamond", a=5.43)
    return Structure.from_ase(atoms)

def test_apply_vacancy(base_structure):
    strategy = DefectStrategy()
    results = list(strategy.apply(base_structure, defect_type="vacancy"))

    assert len(results) > 0
    for s in results:
        assert len(s.symbols) == len(base_structure.symbols) - 1

def test_apply_interstitial(base_structure):
    strategy = DefectStrategy()
    results = list(strategy.apply(base_structure, defect_type="interstitial"))

    assert len(results) > 0
    for s in results:
        assert len(s.symbols) == len(base_structure.symbols) + 1

def test_apply_antisite():
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64)
    structure = Structure.from_ase(atoms)

    strategy = DefectStrategy()
    results = list(strategy.apply(structure, defect_type="antisite"))

    assert len(results) > 0
    for s in results:
        assert len(s.symbols) == len(structure.symbols)
        assert s.symbols != structure.symbols

import pytest
import ase.build
from mlip_autopipec.domain_models.structure import Structure
# defects.py will be created in Phase 3
from mlip_autopipec.physics.structure_gen.defects import DefectStrategy

@pytest.fixture
def base_structure():
    atoms = ase.build.bulk("Si", "diamond", a=5.43)
    return Structure.from_ase(atoms)

def test_apply_vacancy(base_structure):
    strategy = DefectStrategy()
    # Assuming apply takes structure and defect type
    results = strategy.apply(base_structure, defect_type="vacancy")

    assert isinstance(results, list)
    assert len(results) > 0
    for s in results:
        # Vacancy should remove 1 atom
        assert len(s.symbols) == len(base_structure.symbols) - 1

def test_apply_interstitial(base_structure):
    strategy = DefectStrategy()
    results = strategy.apply(base_structure, defect_type="interstitial")

    assert isinstance(results, list)
    assert len(results) > 0
    for s in results:
        # Interstitial should add 1 atom
        assert len(s.symbols) == len(base_structure.symbols) + 1

def test_apply_antisite():
    # Need binary compound for antisite
    atoms = ase.build.bulk("NaCl", "rocksalt", a=5.64)
    structure = Structure.from_ase(atoms)

    strategy = DefectStrategy()
    results = strategy.apply(structure, defect_type="antisite")

    assert isinstance(results, list)
    assert len(results) > 0
    for s in results:
        # Number of atoms same
        assert len(s.symbols) == len(structure.symbols)
        # But symbols should differ at some position
        assert s.symbols != structure.symbols

import pytest
import ase.build
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.strain import StrainStrategy

@pytest.fixture
def base_structure():
    atoms = ase.build.bulk("Al", "fcc", a=4.05)
    return Structure.from_ase(atoms)

def test_apply_uniaxial(base_structure):
    strategy = StrainStrategy()
    results = list(strategy.apply(base_structure, strain_type="uniaxial", magnitude=0.05))

    assert len(results) > 0

    original_cell = base_structure.cell

    for s in results:
        assert not np.allclose(s.cell, original_cell)
        vol_diff = np.abs(np.linalg.det(s.cell) - np.linalg.det(original_cell))
        assert vol_diff > 1e-5

def test_apply_shear(base_structure):
    strategy = StrainStrategy()
    results = list(strategy.apply(base_structure, strain_type="shear", magnitude=0.05))

    assert len(results) > 0

    original_cell = base_structure.cell
    for s in results:
        assert not np.allclose(s.cell, original_cell)

def test_apply_rattle(base_structure):
    strategy = StrainStrategy()
    results = list(strategy.apply(base_structure, strain_type="rattle", magnitude=0.1))

    assert len(results) > 0

    for s in results:
        assert np.allclose(s.cell, base_structure.cell)
        assert not np.allclose(s.positions, base_structure.positions)

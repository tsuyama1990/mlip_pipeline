import numpy as np
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
from mlip_autopipec.domain_models.structure import Structure


def test_build_bulk():
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43)

    assert isinstance(structure, Structure)
    assert structure.get_chemical_formula() == "Si2"  # Primitive cell
    assert structure.pbc == (True, True, True)


def test_build_bulk_cubic():
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43, cubic=True)

    assert structure.get_chemical_formula() == "Si8"  # Conventional cubic cell


def test_apply_rattle():
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43)
    original_positions = structure.positions.copy()

    # Apply rattle
    rattled_structure = builder.apply_rattle(structure, stdev=0.1, seed=42)

    # Check positions changed
    assert not np.allclose(structure.positions, rattled_structure.positions)
    assert np.allclose(original_positions, structure.positions) # Ensure original not modified (immutability check if applicable, or just copy check)

    # Check consistency
    assert np.allclose(structure.cell, rattled_structure.cell)
    assert structure.symbols == rattled_structure.symbols


def test_apply_rattle_deterministic():
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43)

    rattled1 = builder.apply_rattle(structure, stdev=0.1, seed=42)
    rattled2 = builder.apply_rattle(structure, stdev=0.1, seed=42)

    assert np.allclose(rattled1.positions, rattled2.positions)

def test_apply_strain():
    builder = StructureBuilder()
    structure = builder.build_bulk("Si", "diamond", 5.43)
    original_cell = structure.cell.copy()

    # Apply 1% strain in x
    strain = np.zeros((3, 3))
    strain[0, 0] = 0.01

    strained_structure = builder.apply_strain(structure, strain)

    # Expected cell
    expected_cell = original_cell @ (np.eye(3) + strain)

    assert np.allclose(strained_structure.cell, expected_cell)
    # Positions should scale? ASE set_cell(scale_atoms=True) is used.
    # Relative positions (fractional) should stay same, but cartesian should change.
    # Since Si diamond starts at 0,0,0, that one stays. Others move.
    assert not np.allclose(structure.positions, strained_structure.positions)

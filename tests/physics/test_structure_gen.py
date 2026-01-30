import numpy as np
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder


def test_build_bulk():
    builder = StructureBuilder()
    struct = builder.build_bulk("Si", "diamond", 5.43)
    assert isinstance(struct, Structure)
    assert "Si" in struct.symbols
    assert len(struct.positions) > 0


def test_apply_rattle():
    builder = StructureBuilder()
    struct = builder.build_bulk("Si", "diamond", 5.43)
    orig_pos = struct.positions.copy()

    rattled = builder.apply_rattle(struct, stdev=0.1, seed=42)
    assert not np.allclose(orig_pos, rattled.positions)
    assert np.allclose(struct.cell, rattled.cell)

def test_apply_strain():
    builder = StructureBuilder()
    struct = builder.build_bulk("Si", "diamond", 5.43)
    orig_cell = struct.cell.copy()

    # Apply 10% strain in x
    strain = np.zeros((3,3))
    strain[0,0] = 0.1

    strained = builder.apply_strain(struct, strain)

    # Cell is row vectors in ASE/Structure?
    # Structure validate_cell_shape checks 3x3.
    # ASE atoms.get_cell() returns 3x3.
    # Logic: new_cell = cell @ (I + epsilon)
    # If epsilon is diagonal 0.1, then cell[0] -> cell[0] * 1.1

    expected_cell = np.dot(orig_cell, np.eye(3) + strain)
    assert np.allclose(strained.cell, expected_cell)

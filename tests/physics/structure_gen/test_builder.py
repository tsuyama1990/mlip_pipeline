import numpy as np

from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder


def test_build_bulk_si() -> None:
    """Test building bulk Silicon."""
    builder = StructureBuilder()
    struct = builder.build_bulk(element="Si", crystal_structure="diamond", lattice_constant=5.43)

    assert isinstance(struct, Structure)
    assert "Si" in struct.symbols
    assert len(struct.positions) > 0
    # Diamond structure has 2 atoms in primitive cell
    assert len(struct.positions) == 2

def test_apply_rattle() -> None:
    """Test applying rattle."""
    builder = StructureBuilder(seed=42)
    struct_orig = builder.build_bulk("Cu", "fcc", 3.6)

    struct_rattled = builder.apply_rattle(struct_orig, stdev=0.1)

    # Check positions changed
    pos_orig = struct_orig.positions
    pos_new = struct_rattled.positions
    assert not np.allclose(pos_orig, pos_new)

    # Check cell did NOT change
    assert np.allclose(struct_orig.cell, struct_rattled.cell)

def test_apply_strain() -> None:
    """Test applying strain."""
    builder = StructureBuilder()
    struct_orig = builder.build_bulk("Cu", "fcc", 3.6)

    # Apply 10% expansion in x
    strain = np.eye(3)
    strain[0, 0] = 1.1

    struct_strained = builder.apply_strain(struct_orig, strain)

    # Cell should be different
    assert not np.allclose(struct_orig.cell, struct_strained.cell)
    assert np.isclose(struct_strained.cell[0, 0], struct_orig.cell[0, 0] * 1.1)

def test_reproducibility() -> None:
    """Test that seed works."""
    b1 = StructureBuilder(seed=123)
    s1 = b1.build_bulk("Al", "fcc", 4.05)
    s1_r = b1.apply_rattle(s1, 0.1)

    b2 = StructureBuilder(seed=123)
    s2 = b2.build_bulk("Al", "fcc", 4.05)
    s2_r = b2.apply_rattle(s2, 0.1)

    assert np.allclose(s1_r.positions, s2_r.positions)

import pytest
import numpy as np
from mlip_autopipec.domain_models.structure import Structure
# We will import StructureBuilder from physics.structure_gen.builder later

def test_build_bulk_si():
    from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
    builder = StructureBuilder()

    # Test Silicon Diamond
    struct = builder.build_bulk("Si", "diamond", 5.43)

    assert isinstance(struct, Structure)
    assert len(struct.positions) == 8  # 8 atoms in conventional cubic diamond
    assert struct.symbols == ["Si"] * 8
    # Assert cell is roughly 5.43^3
    assert np.allclose(struct.cell, np.eye(3) * 5.43)

def test_apply_rattle():
    from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
    builder = StructureBuilder()

    # Create initial structure
    initial_struct = builder.build_bulk("Cu", "fcc", 3.6)
    initial_pos = initial_struct.positions.copy()

    # Apply rattle
    rattled = builder.apply_rattle(initial_struct, stdev=0.1, seed=42)

    assert isinstance(rattled, Structure)
    assert not np.allclose(initial_pos, rattled.positions)
    assert np.allclose(initial_struct.cell, rattled.cell) # Cell shouldn't change

def test_rattle_validation_failure():
    """Test that applying too much rattle causing overlaps raises ValueError."""
    from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
    builder = StructureBuilder()

    # Create a small cell with many atoms or just force a huge rattle
    # Using a small lattice constant to start close might help
    # Diamond at 1.0A: NN dist is ~0.433 A < 0.5 A default limit
    struct = builder.build_bulk("Si", "diamond", 1.0)

    # Apply huge rattle to guarantee overlap or just rely on dense start
    # Actually, dense start might pass if it's perfect crystal?
    # Diamond at 2.0A: NN dist is sqrt(3)/4 * 2.0 = 0.866 A.
    # If min_dist is 0.5, it might pass.
    # Let's try 1.0A lattice: NN = 0.433 A. Should fail immediately even with 0 rattle if we checked.
    # But we check in apply_rattle.

    with pytest.raises(ValueError, match="Atoms are too close"):
         builder.apply_rattle(struct, stdev=0.0)

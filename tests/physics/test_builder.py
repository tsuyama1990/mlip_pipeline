import numpy as np
import pytest
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder


@pytest.fixture
def builder():
    return StructureBuilder()


def test_build_bulk(builder):
    struct = builder.build_bulk("Si", "diamond", 5.43, cubic=True)
    assert isinstance(struct, Structure)
    assert struct.symbols[0] == "Si"
    # Diamond cubic has 8 atoms
    assert len(struct.positions) == 8


def test_apply_rattle(builder, sample_ase_atoms):
    struct = Structure.from_ase(sample_ase_atoms)
    rattled = builder.apply_rattle(struct, stdev=0.1, seed=42)
    assert not np.allclose(struct.positions, rattled.positions)
    assert np.allclose(struct.cell, rattled.cell)

    # Check determinism
    rattled2 = builder.apply_rattle(struct, stdev=0.1, seed=42)
    assert np.allclose(rattled.positions, rattled2.positions)

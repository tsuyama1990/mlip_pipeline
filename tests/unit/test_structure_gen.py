import numpy as np
import pytest
from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.physics.structure_gen.builder import StructureBuilder
from mlip_autopipec.physics.structure_gen.generator import StructureGenFactory


def test_builder_build_bulk_si() -> None:
    builder = StructureBuilder()
    struct = builder.build_bulk("Si", "diamond", 5.43)
    assert len(struct.positions) == 8
    assert struct.symbols == ["Si"] * 8
    assert np.allclose(struct.cell, np.eye(3) * 5.43)


def test_builder_apply_rattle() -> None:
    builder = StructureBuilder()
    initial_struct = builder.build_bulk("Cu", "fcc", 3.6)
    initial_pos = initial_struct.positions.copy()
    rattled = builder.apply_rattle(initial_struct, stdev=0.1, seed=42)
    assert not np.allclose(initial_pos, rattled.positions)


def test_builder_rattle_validation_failure() -> None:
    """Test that applying too much rattle causing overlaps raises ValueError."""
    builder = StructureBuilder()
    # 1.0A lattice constant for diamond results in NN distance < 0.5A
    struct = builder.build_bulk("Si", "diamond", 1.0)
    with pytest.raises(ValueError, match="Atoms are too close"):
        builder.apply_rattle(struct, stdev=0.0)


def test_strategy_bulk_gen() -> None:
    config = StructureGenConfig(
        strategy="bulk",
        element="Si",
        crystal_structure="diamond",
        lattice_constant=5.43,
        rattle_stdev=0.0,
        supercell=(1, 1, 1),
    )
    generator = StructureGenFactory.get_generator(config)
    struct = generator.generate(config)

    assert len(struct.positions) == 8
    assert struct.symbols == ["Si"] * 8


def test_strategy_bulk_supercell_rattle() -> None:
    config = StructureGenConfig(
        strategy="bulk",
        element="Cu",
        crystal_structure="fcc",
        lattice_constant=3.61,
        rattle_stdev=0.1,
        supercell=(2, 1, 1),
    )
    generator = StructureGenFactory.get_generator(config)
    struct = generator.generate(config)

    # FCC unit cell has 4 atoms. 2x1x1 supercell -> 8 atoms.
    assert len(struct.positions) == 8
    assert struct.symbols == ["Cu"] * 8

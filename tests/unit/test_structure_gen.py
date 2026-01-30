from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator


def test_structure_generator_bulk() -> None:
    """Test generating a bulk structure."""
    config = StructureGenConfig(
        element="Si",
        crystal_structure="diamond",
        lattice_constant=5.43,
        supercell=(1, 1, 1)
    )
    gen = StructureGenerator()
    s = gen.generate(config)
    assert s.symbols[0] == "Si"
    assert len(s.symbols) == 8 # Diamond conventional cell has 8 atoms
    assert s.pbc == (True, True, True)

def test_structure_generator_supercell() -> None:
    """Test generating a supercell."""
    config = StructureGenConfig(
        element="Si",
        crystal_structure="diamond",
        lattice_constant=5.43,
        supercell=(2, 1, 1)
    )
    gen = StructureGenerator()
    s = gen.generate(config)
    assert len(s.symbols) == 16

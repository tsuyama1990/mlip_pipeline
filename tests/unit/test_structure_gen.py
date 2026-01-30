from unittest.mock import MagicMock, patch
import ase

from mlip_autopipec.domain_models.config import StructureGenConfig
from mlip_autopipec.domain_models.structure import Structure
from mlip_autopipec.modules.structure_gen.generator import StructureGenerator

def test_generator_initialization() -> None:
    config = StructureGenConfig(element="Si")
    gen = StructureGenerator(config)
    assert gen.config == config

@patch("mlip_autopipec.modules.structure_gen.strategies.ColdStartStrategy.generate")
def test_build_initial_structure(mock_generate: MagicMock) -> None:
    # Setup mock return value
    mock_atoms = ase.Atoms("Si2", positions=[[0,0,0], [1,1,1]], cell=[5,5,5], pbc=True)
    mock_generate.return_value = Structure.from_ase(mock_atoms)

    config = StructureGenConfig(element="Si")
    gen = StructureGenerator(config)

    s = gen.build_initial_structure()

    assert isinstance(s, Structure)
    assert s.symbols == ["Si", "Si"]
    mock_generate.assert_called_once()

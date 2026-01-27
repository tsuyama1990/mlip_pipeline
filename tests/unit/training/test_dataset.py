from unittest.mock import MagicMock

import pytest
from ase import Atoms

from mlip_autopipec.training.dataset import DatasetBuilder


@pytest.fixture
def mock_db_manager():
    return MagicMock()

def test_export_atoms_iterable(mock_db_manager, tmp_path):
    builder = DatasetBuilder(mock_db_manager)
    atoms_list = [Atoms("H"), Atoms("He")]

    # Verify it accepts generator
    def atom_gen():
        yield from atoms_list

    output = tmp_path / "test.xyz"
    builder.export_atoms(atom_gen(), output)

    assert output.exists()
    # verify content roughly
    assert "H" in output.read_text()
    assert "He" in output.read_text()

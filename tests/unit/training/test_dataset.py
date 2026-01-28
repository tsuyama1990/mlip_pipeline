from unittest.mock import MagicMock

import pytest
from ase import Atoms

from mlip_autopipec.training.dataset import DatasetBuilder


@pytest.fixture
def mock_db_manager():
    return MagicMock()


def test_dataset_export_streaming(mock_db_manager, tmp_path):
    builder = DatasetBuilder(mock_db_manager)

    # Mock streaming return
    atoms_list = [Atoms("H"), Atoms("He")]

    def atom_gen(*args, **kwargs):
        yield from atoms_list

    mock_db_manager.select.side_effect = atom_gen

    output_path = tmp_path / "data.xyz"
    builder.export(output_path, query="all")

    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "test.xyz").exists()

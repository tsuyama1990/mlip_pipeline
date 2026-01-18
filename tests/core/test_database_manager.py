from unittest.mock import Mock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.data_models.dft_models import DFTResult


@pytest.fixture
def mock_db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def db_manager(mock_db_path):
    return DatabaseManager(mock_db_path)


def test_database_initialization(db_manager, mock_db_path):
    with patch("ase.db.connect") as mock_connect:
        db_manager.initialize()
        mock_connect.assert_called_once_with(str(mock_db_path))
        assert mock_db_path.parent.exists()


def test_get_training_data(db_manager):
    # Mock row data that matches TrainingData schema
    mock_row = Mock()
    mock_row.data = {"energy": -10.5, "forces": [[0.1, 0.2, 0.3]]}
    mock_atom = Atoms("H")
    mock_row.toatoms.return_value = mock_atom

    with patch("ase.db.connect") as mock_connect:
        mock_db_instance = Mock()
        mock_db_instance.select.return_value = [mock_row]
        mock_connect.return_value = mock_db_instance

        db_manager.initialize()
        atoms_list = db_manager.get_training_data()

        assert len(atoms_list) == 1
        assert atoms_list[0].info["energy"] == -10.5
        np.testing.assert_array_equal(atoms_list[0].arrays["forces"], np.array([[0.1, 0.2, 0.3]]))


def test_save_dft_result(db_manager):
    atoms = Atoms("H")
    result = DFTResult(
        uid="123",
        energy=-13.6,
        forces=[[0.0, 0.0, 0.0]],
        stress=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        succeeded=True,
        wall_time=10.0,
        parameters={},
        final_mixing_beta=0.7,
    )
    metadata = {"key": "value", "force_mask": [[1.0, 1.0, 1.0]]}

    with patch("ase.db.connect") as mock_connect:
        mock_db_instance = Mock()
        mock_connect.return_value = mock_db_instance
        db_manager.initialize()

        db_manager.save_dft_result(atoms, result, metadata)

        # Verify atoms object updated
        assert atoms.info["energy"] == -13.6
        assert atoms.info["key"] == "value"
        np.testing.assert_array_equal(atoms.arrays["force_mask"], np.array([[1.0, 1.0, 1.0]]))

        # Verify write call
        mock_db_instance.write.assert_called_once()

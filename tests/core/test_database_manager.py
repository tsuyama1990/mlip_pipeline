from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.exceptions import DatabaseError
from mlip_autopipec.data_models.dft_models import DFTResult


@pytest.fixture
def mock_ase_db(tmp_path):
    db_path = tmp_path / "test.db"
    return DatabaseManager(db_path)


def test_db_manager_initialization(mock_ase_db):
    mock_ase_db.initialize()
    assert mock_ase_db._connection is not None
    assert mock_ase_db.db_path.exists()


def test_db_manager_save_retrieve_candidate(mock_ase_db):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = {"status": "pending", "generation": 0}

    mock_ase_db.save_candidate(atoms, metadata)

    retrieved = mock_ase_db.get_atoms("status=pending")
    assert len(retrieved) == 1
    assert retrieved[0].info["status"] == "pending"
    assert retrieved[0].info["generation"] == 0


def test_db_manager_count(mock_ase_db):
    atoms = Atoms("H")
    mock_ase_db.save_candidate(atoms, {"status": "pending"})
    mock_ase_db.save_candidate(atoms, {"status": "training"})

    assert mock_ase_db.count() == 2
    assert mock_ase_db.count("status=pending") == 1


def test_db_manager_connection_error(tmp_path):
    # Test initialization failure
    db = DatabaseManager(tmp_path / "non_existent_dir/db.db") # Parent dir doesn't exist
    # Actually initialize makes dirs, so let's mock connect to raise

    with patch("ase.db.connect", side_effect=Exception("Connection Failed")):
        with pytest.raises(DatabaseError, match="Failed to initialize"):
            db.initialize()


def test_db_manager_operation_without_init(tmp_path):
    # Should auto-initialize
    db = DatabaseManager(tmp_path / "test.db")
    assert db.count() == 0
    assert db._connection is not None

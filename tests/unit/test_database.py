import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config.models import MinimalConfig, SystemConfig, TargetSystem
from mlip_autopipec.exceptions import DatabaseError
from mlip_autopipec.orchestration.database import DatabaseManager


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"

def test_database_init(db_path):
    with DatabaseManager(db_path) as db:
        assert db_path.exists()
        assert db.count() == 0

def test_add_structure(db_path):
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = {"status": "pending", "generation": 0}

    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, metadata)
        assert uid == 1
        assert db.count() == 1

        entries = list(db.get_entries())
        assert entries[0][0] == 1
        assert len(entries[0][1]) == 2
        assert entries[0][1].info["status"] == "pending"

def test_update_status(db_path):
    atoms = Atoms('H')
    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, {"status": "pending"})
        db.update_status(uid, "running")

        entries = list(db.get_entries())
        assert entries[0][1].info["status"] == "running"

def test_validate_atoms_nan(db_path):
    atoms = Atoms('H', positions=[[float('nan'), 0, 0]])
    with DatabaseManager(db_path) as db:
        with pytest.raises(DatabaseError) as exc:
            db.add_structure(atoms, {})
        assert "Invalid Atoms object" in str(exc.value)

def test_validate_atoms_zero_cell_pbc(db_path):
    atoms = Atoms('H', cell=[0,0,0], pbc=True)
    with DatabaseManager(db_path) as db:
        with pytest.raises(DatabaseError) as exc:
            db.add_structure(atoms, {})
        assert "zero cell volume" in str(exc.value)

def test_count_kwargs(db_path):
    atoms = Atoms('H')
    with DatabaseManager(db_path) as db:
        db.add_structure(atoms, {"status": "pending"})
        db.add_structure(atoms, {"status": "completed"})

        assert db.count(status="pending") == 1
        assert db.count(status="completed") == 1

def test_update_metadata(db_path):
    atoms = Atoms('H')
    with DatabaseManager(db_path) as db:
        uid = db.add_structure(atoms, {"status": "pending"})
        db.update_metadata(uid, {"new_key": "value"})

        entries = list(db.get_entries())
        assert entries[0][1].info["new_key"] == "value"

def test_get_atoms(db_path):
    atoms = Atoms('H')
    with DatabaseManager(db_path) as db:
        db.add_structure(atoms, {"status": "pending", "foo": "bar"})

        fetched = list(db.get_atoms(status="pending"))
        assert len(fetched) == 1
        assert fetched[0].info["foo"] == "bar"
        assert fetched[0].info["status"] == "pending"

def test_save_candidate(db_path):
    atoms = Atoms('H')
    with DatabaseManager(db_path) as db:
        db.save_candidate(atoms, {"status": "pending", "source": "random"})
        assert db.count() == 1
        atoms_list = list(db.get_atoms())
        assert atoms_list[0].info["source"] == "random"

def test_save_dft_result(db_path):
    from pydantic import BaseModel
    class MockResult(BaseModel):
        energy: float = -10.0
        forces: list = [[0.0, 0.0, 0.0]]
        stress: list = [0.0]*6

    atoms = Atoms('H')
    result = MockResult()
    with DatabaseManager(db_path) as db:
        db.save_dft_result(atoms, result, {"status": "completed"})

        assert db.count() == 1
        saved = next(iter(db.get_atoms()))
        assert saved.info["energy"] == -10.0
        # Forces are saved in 'data' blob, which is merged into info by our get_atoms
        assert np.allclose(saved.info["forces"], [[0.0, 0.0, 0.0]])
        assert saved.info["status"] == "completed"

def test_system_config(db_path):
    sys_conf = SystemConfig(
        target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}),
        minimal=MinimalConfig(target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}))
    )

    with DatabaseManager(db_path) as db:
        db.set_system_config(sys_conf)

        # Re-open to check persistence (ase.db metadata is stored in file)

    with DatabaseManager(db_path) as db:
        loaded = db.get_system_config()
        assert loaded.target_system.elements == ["Fe"]

# Error Handling Tests

def test_connect_os_error(db_path):
    with patch("ase.db.connect", side_effect=OSError("Disk full")):
        db = DatabaseManager(db_path)
        with pytest.raises(DatabaseError) as exc:
            db.connector.connect()
        assert "FileSystem error" in str(exc.value)

def test_connect_sqlite_error(db_path):
    with patch("ase.db.connect", side_effect=sqlite3.DatabaseError("Corrupt")):
        db = DatabaseManager(db_path)
        with pytest.raises(DatabaseError) as exc:
            db.connector.connect()
        assert "not a valid SQLite database" in str(exc.value)

def test_add_structure_key_error(db_path):
    atoms = Atoms('H')
    # Mock _connection.write to raise KeyError
    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.write.side_effect = KeyError("bad key")

        with pytest.raises(DatabaseError) as exc:
            db.add_structure(atoms, {})
        assert "Invalid key" in str(exc.value)

def test_update_status_key_error(db_path):
    with DatabaseManager(db_path) as db:
        # ID 999 does not exist
        with pytest.raises(DatabaseError) as exc:
            db.update_status(999, "running")
        assert "Failed to update status" in str(exc.value)

def test_get_atoms_error(db_path):
    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.select.side_effect = Exception("Select failed")

        with pytest.raises(DatabaseError) as exc:
            list(db.get_atoms()) # Must iterate to trigger error
        assert "Failed to select atoms" in str(exc.value)

def test_get_entries_error(db_path):
    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.select.side_effect = Exception("Select failed")

        with pytest.raises(DatabaseError) as exc:
            list(db.get_entries()) # Must iterate to trigger error
        assert "Failed to select entries" in str(exc.value)

def test_count_error(db_path):
    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.count.side_effect = Exception("Count failed")

        with pytest.raises(DatabaseError) as exc:
            db.count()
        assert "Failed to count rows" in str(exc.value)

def test_update_metadata_error(db_path):
    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.update.side_effect = Exception("Update failed")

        with pytest.raises(DatabaseError) as exc:
            db.update_metadata(1, {})
        assert "Failed to update metadata" in str(exc.value)

def test_get_system_config_invalid(db_path):
    with DatabaseManager(db_path) as db:
        # Inject invalid metadata directly into the mock or connection if possible
        # ase.db metadata is a dict.
        db.connector.connect().metadata = {"target_system": "Not A Dict"}

        with pytest.raises(DatabaseError) as exc:
            db.get_system_config()
        assert "Stored SystemConfig is invalid" in str(exc.value)

def test_save_dft_result_error(db_path):
    atoms = Atoms('H')
    from pydantic import BaseModel
    class MockResult(BaseModel):
        energy: float = -10.0

    with DatabaseManager(db_path) as db:
        db.connector._connection = MagicMock()
        db.connector._connection.write.side_effect = Exception("Write failed")

        with pytest.raises(DatabaseError) as exc:
            db.save_dft_result(atoms, MockResult(), {})
        assert "Failed to save DFT result" in str(exc.value)

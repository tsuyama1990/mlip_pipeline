import pytest
from ase import Atoms
from mlip_autopipec.core.database import DatabaseManager

def test_update_metadata(tmp_path):
    """Test updating and retrieving metadata."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    atoms = Atoms('H')
    db.save_candidate(atoms, {"status": "pending"})

    entries = db.get_entries()
    id = entries[0][0]

    db.update_metadata(id, {"test_val": 1.23})

    # Verify row directly
    row = db._connection.get(id=id)
    assert row.test_val == 1.23

    # Verify toatoms via get_atoms
    atoms_list = db.get_atoms()
    assert "test_val" in atoms_list[0].info
    assert atoms_list[0].info["test_val"] == 1.23

def test_get_entries(tmp_path):
    """Test get_entries returns (id, atoms) tuples."""
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)

    atoms = Atoms('H')
    db.save_candidate(atoms, {"status": "pending"})

    entries = db.get_entries()
    assert len(entries) == 1
    assert isinstance(entries[0], tuple)
    assert isinstance(entries[0][0], int)
    assert isinstance(entries[0][1], Atoms)

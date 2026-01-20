from ase import Atoms

from mlip_autopipec.core.database import DatabaseManager


def test_database_initialization(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()
    assert db_path.exists()

def test_add_calculation(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()

    atoms = Atoms("H2O")
    atoms.info["energy"] = -10.5

    row_id = db.add_calculation(atoms, {"calculation_type": "scf"})
    assert row_id == 1
    assert db.count() == 1

def test_get_pending_calculations(tmp_path):
    db_path = tmp_path / "test.db"
    db = DatabaseManager(db_path)
    db.initialize()

    atoms = Atoms("Si")
    db.save_candidate(atoms, {"status": "pending", "generation": 0})

    pending = db.get_pending_calculations()
    assert len(pending) == 1
    assert pending[0].get_chemical_symbols()[0] == "Si"

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from ase import Atoms
from mlip_autopipec.orchestration.database import DatabaseManager, DatabaseError

@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    return DatabaseManager(tmp_path / "test.db")

def test_select_entries_generator(db_manager: DatabaseManager) -> None:
    # Mock connection and select
    mock_conn = MagicMock()
    # Create fake rows
    row1 = MagicMock()
    row1.id = 1
    row1.toatoms.return_value = Atoms("H")
    row1.key_value_pairs = {}
    row1.data = {}

    row2 = MagicMock()
    row2.id = 2
    row2.toatoms.return_value = Atoms("He")
    row2.key_value_pairs = {}
    row2.data = {}

    def row_generator():
        yield row1
        yield row2

    # select returns generator of rows
    mock_conn.select.return_value = row_generator()

    with patch.object(db_manager, "connector") as mock_connector:
        mock_connector.connect.return_value = mock_conn

        gen = db_manager.select_entries()

        # Consuming generator one by one
        item1 = next(gen)
        assert item1[0] == 1
        assert item1[1].get_chemical_formula() == "H"

        item2 = next(gen)
        assert item2[0] == 2
        assert item2[1].get_chemical_formula() == "He"

        with pytest.raises(StopIteration):
            next(gen)

def test_add_structure_validation_error(db_manager: DatabaseManager) -> None:
    atoms = Atoms("H")
    atoms.positions[0] = [float("nan"), 0, 0]

    with pytest.raises(DatabaseError, match="Invalid Atoms object"):
        db_manager.add_structure(atoms, {})

def test_select_entries_error_handling(db_manager: DatabaseManager) -> None:
    mock_conn = MagicMock()
    mock_conn.select.side_effect = Exception("DB Fail")

    with patch.object(db_manager, "connector") as mock_connector:
        mock_connector.connect.return_value = mock_conn

        gen = db_manager.select_entries()
        with pytest.raises(DatabaseError, match="Failed to select entries"):
            next(gen)

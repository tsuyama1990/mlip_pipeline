from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.exceptions import DatabaseError
from mlip_autopipec.orchestration.database import DatabaseManager


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

    # select returns generator of rows
    mock_conn.select.return_value = iter([row1, row2])

    with patch.object(db_manager, "connector") as mock_connector:
        mock_connector.connect.return_value = mock_conn

        gen = db_manager.select_entries()

        # Consuming generator
        items = list(gen)
        assert len(items) == 2
        assert items[0][0] == 1
        assert items[0][1].get_chemical_formula() == "H"
        assert items[1][0] == 2
        assert items[1][1].get_chemical_formula() == "He"

def test_add_structure_validation_error(db_manager: DatabaseManager) -> None:
    atoms = Atoms("H")
    atoms.positions[0] = [float("nan"), 0, 0]

    with pytest.raises(DatabaseError, match="Invalid Atoms object"):
        db_manager.add_structure(atoms, {})

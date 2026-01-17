from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase.db.core import Database

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.exceptions import DatabaseError


def test_database_manager_initialize_fail_permissions(tmp_path):
    """Test that initialize handles permission errors gracefully (log only)."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    # Mock ase.db.connect
    with patch("ase.db.connect") as mock_connect:
        mock_db = MagicMock(spec=Database)
        mock_connect.return_value = mock_db

        # Mock chmod to raise OSError
        with patch.object(Path, "chmod", side_effect=OSError("Permission denied")):
            manager.initialize()

        mock_db.count.assert_called_once()  # Should still proceed


def test_database_manager_initialize_fail_connect(tmp_path):
    """Test failure during connection."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    with patch("ase.db.connect", side_effect=Exception("Connection Error")):
        with pytest.raises(DatabaseError) as excinfo:
            manager.initialize()
        assert "Failed to initialize database" in str(excinfo.value)


def test_get_system_config_invalid_metadata(tmp_path):
    """Test retrieving invalid config raises DatabaseError."""
    db_path = tmp_path / "test.db"
    manager = DatabaseManager(db_path)

    with patch("ase.db.connect") as mock_connect:
        mock_db = MagicMock(spec=Database)
        # Return invalid metadata
        mock_db.metadata = {"minimal": "invalid"}
        mock_connect.return_value = mock_db

        with pytest.raises(DatabaseError) as excinfo:
            manager.get_system_config()
        assert "valid SystemConfig" in str(excinfo.value)

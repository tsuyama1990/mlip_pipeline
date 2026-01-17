from unittest.mock import patch

import pytest

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.exceptions import DatabaseError


def test_initialize_security(tmp_path):
    """Test that initialize sets file permissions."""
    db_path = tmp_path / "secure.db"
    manager = DatabaseManager(db_path)

    # Mock ase.db.connect but allow filesystem side effects
    with patch("ase.db.connect") as mock_connect:
        mock_connect.return_value.count.return_value = 0
        # Create dummy file to check permissions on
        db_path.touch()
        manager.initialize()

    # Check permissions (only works on POSIX)
    import os

    if os.name == "posix":
        mode = db_path.stat().st_mode
        # Check for 600 (rw-------)
        # Note: Depending on umask, it might be different, but we explicitly set it.
        # stat.S_IMODE gets the last 12 bits
        import stat

        assert stat.S_IMODE(mode) == 0o600


def test_get_metadata_error_handling(tmp_path):
    """Test that get_metadata raises DatabaseError if connection fails."""
    db_path = tmp_path / "error.db"
    manager = DatabaseManager(db_path)

    with patch("ase.db.connect", side_effect=Exception("Connection Failed")):
        with pytest.raises(DatabaseError) as excinfo:
            manager.get_metadata()
        assert "Failed to initialize" in str(excinfo.value)

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.config.models import SystemConfig, MinimalConfig, Resources, TargetSystem

@pytest.fixture
def mock_system_config(tmp_path):
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )
    return SystemConfig(
        minimal=minimal,
        working_dir=tmp_path,
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "log.txt"
    )

def test_database_initialization(mock_system_config):
    """Test that database initializes and writes metadata."""
    with patch("ase.db.connect") as mock_connect:
        mock_db = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_db

        # Don't set metadata to a dict, let it be a Mock
        # So db.metadata returns a Mock object.

        db_manager = DatabaseManager(mock_system_config.db_path)
        db_manager.initialize(mock_system_config)

        mock_connect.assert_called_with(mock_system_config.db_path)

        # Verify metadata.update was called
        mock_db.metadata.update.assert_called_once()

        # Check arguments to update
        call_args = mock_db.metadata.update.call_args[0][0]
        assert "system_config" in call_args
        assert call_args["system_config"]["minimal"]["project_name"] == "Test"

def test_get_metadata(mock_system_config):
    """Test retrieving metadata."""
    mock_system_config.db_path.touch()

    with patch("ase.db.connect") as mock_connect:
        mock_db = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_db

        # Here we return a dict when accessing metadata
        mock_db.metadata = {"system_config": mock_system_config.model_dump(mode='json')}

        db_manager = DatabaseManager(mock_system_config.db_path)
        meta = db_manager.get_metadata()

        assert meta["system_config"]["minimal"]["project_name"] == "Test"

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.config.models import (
    CheckpointState,
    SystemConfig,
    TrainingRunMetrics,
    WorkflowConfig,
)

runner = CliRunner()

from mlip_autopipec.config.models import TrainingConfig


# Helper to create valid system config
def create_system_config():
    return SystemConfig(
        project_name="Integration Test Project",
        run_uuid=uuid.uuid4(),
        workflow_config=WorkflowConfig(checkpoint_filename="checkpoint.json"),
        training_config=TrainingConfig(data_source_db=Path("mlip_database.db")),
    )

@pytest.fixture
def populated_project(tmp_path):
    # Create checkpoint
    metrics = [
        TrainingRunMetrics(
            generation=1, num_structures=100, rmse_forces=0.1, rmse_energy_per_atom=0.01
        ),
    ]
    state = CheckpointState(
        run_uuid=uuid.uuid4(),
        system_config=create_system_config(),
        active_learning_generation=1,
        training_history=metrics,
    )
    (tmp_path / "checkpoint.json").write_text(state.model_dump_json())

    # Create DB using real ASE DB
    from ase import Atoms
    from ase.db import connect
    db_path = tmp_path / "mlip_database.db"
    with connect(db_path) as db:
         db.write(Atoms('H2'), data={'config_type': 'initial'})
         db.write(Atoms('O2'), data={'config_type': 'active_learning'})

    return tmp_path

def test_status_command(populated_project):
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["status", str(populated_project)])

        # If command not found, exit code is 2.
        if result.exit_code == 2:
            pytest.fail("Command 'status' not found in app.")

        if result.exit_code != 0:
            print(result.stdout)

        assert result.exit_code == 0
        assert "Dashboard generated at" in result.stdout
        assert (populated_project / "dashboard.html").exists()

        # Check content
        content = (populated_project / "dashboard.html").read_text()
        assert "Integration Test Project" in content

        mock_open.assert_called_once()

def test_status_command_no_open(populated_project):
    with patch("webbrowser.open") as mock_open:
        result = runner.invoke(app, ["status", str(populated_project), "--no-open"])
        if result.exit_code == 2:
             pytest.fail("Command 'status' not found in app.")

        assert result.exit_code == 0
        mock_open.assert_not_called()

def test_status_command_failure(tmp_path):
    """Test status command failure when checkpoint is missing."""
    # Empty dir, no checkpoint
    result = runner.invoke(app, ["status", str(tmp_path)])

    assert result.exit_code != 0
    assert "FILE ERROR" in result.stdout or "FAILURE" in result.stdout

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.models import (
    CheckpointState,
    DashboardData,
    SystemConfig,
    TrainingConfig,
    TrainingRunMetrics,
    WorkflowConfig,
)
from mlip_autopipec.monitoring.dashboard import _gather_data, generate_dashboard


# Helper to create valid system config
def create_system_config():
    return SystemConfig(
        project_name="Test Project",
        run_uuid=uuid.uuid4(),
        workflow_config=WorkflowConfig(checkpoint_filename="checkpoint.json"),
        training_config=TrainingConfig(data_source_db=Path("mlip_database.db")),
    )

@pytest.fixture
def mock_checkpoint_data(tmp_path):
    metrics = [
        TrainingRunMetrics(
            generation=1, num_structures=100, rmse_forces=0.1, rmse_energy_per_atom=0.01
        ),
        TrainingRunMetrics(
            generation=2, num_structures=150, rmse_forces=0.08, rmse_energy_per_atom=0.008
        ),
    ]
    state = CheckpointState(
        run_uuid=uuid.uuid4(),
        system_config=create_system_config(),
        active_learning_generation=2,
        training_history=metrics,
        pending_job_ids=[uuid.uuid4()],
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(state.model_dump_json())
    return state

@patch("mlip_autopipec.monitoring.dashboard.ase_db_connect")
def test_gather_data(mock_connect, mock_checkpoint_data, tmp_path):
    # Setup mock DB
    mock_db = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_db

    # Mock DB select to return counting
    # We assume implementation iterates over rows or uses some method to count
    # Let's mock select returning objects with .data attribute
    row1 = MagicMock()
    row1.data = {"config_type": "initial"}
    row2 = MagicMock()
    row2.data = {"config_type": "active_learning"}
    row3 = MagicMock()
    row3.data = {"config_type": "initial"}

    # Also need to make sure we can get length
    mock_db.select.return_value = [row1, row2, row3]
    mock_db.__len__.return_value = 3

    # Ensure DB file exists
    (tmp_path / "mlip_database.db").touch()

    # Act
    data = _gather_data(tmp_path)

    # Assert
    assert isinstance(data, DashboardData)
    assert data.project_name == "Test Project"
    assert data.current_generation == 2
    assert data.pending_calcs == 1
    assert len(data.training_history) == 2
    assert data.training_history[0].rmse_forces == 0.1
    # Check dataset composition
    assert data.dataset_composition == {"initial": 2, "active_learning": 1}
    # Check completed calcs
    assert data.completed_calcs == 3

@patch("mlip_autopipec.monitoring.dashboard._gather_data")
@patch("mlip_autopipec.monitoring.dashboard._create_plots")
@patch("mlip_autopipec.monitoring.dashboard._render_html")
def test_generate_dashboard(mock_render, mock_plots, mock_gather, tmp_path):
    # Setup
    mock_gather.return_value = MagicMock(spec=DashboardData)
    mock_plots.return_value = {"rmse_plot": "<div></div>"}
    mock_render.return_value = "<html></html>"

    # Act
    output_path = generate_dashboard(tmp_path)

    # Assert
    assert output_path == tmp_path / "dashboard.html"
    assert output_path.exists()
    assert output_path.read_text() == "<html></html>"
    mock_gather.assert_called_once_with(tmp_path)
    mock_plots.assert_called_once()
    mock_render.assert_called_once()

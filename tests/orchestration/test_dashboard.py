from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.dashboard import Dashboard
from mlip_autopipec.orchestration.models import DashboardData


@pytest.fixture
def mock_db_manager():
    return MagicMock(spec=DatabaseManager)

def test_dashboard_update(tmp_path: Path, mock_db_manager) -> None:
    dashboard = Dashboard(tmp_path, mock_db_manager)
    data = DashboardData(
        generations=[0, 1],
        rmse_values=[0.5, 0.4],
        structure_counts=[100, 200],
        status="training"
    )

    dashboard.update(data)

    assert (tmp_path / "dashboard.html").exists()
    content = (tmp_path / "dashboard.html").read_text()
    assert "MLIP-AutoPipe Status" in content
    assert "Current Status: training" in content
    assert "Current Generation: 1" in content
    assert "data:image/png;base64" in content # Plot exists

def test_dashboard_empty_data(tmp_path: Path, mock_db_manager) -> None:
    dashboard = Dashboard(tmp_path, mock_db_manager)
    data = DashboardData(status="idle")

    dashboard.update(data)

    content = (tmp_path / "dashboard.html").read_text()
    assert "No data yet." in content

from unittest.mock import MagicMock

import pytest

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.dashboard import Dashboard
from mlip_autopipec.orchestration.models import DashboardData


@pytest.fixture
def mock_db_manager():
    return MagicMock(spec=DatabaseManager)


@pytest.fixture
def dashboard(tmp_path, mock_db_manager):
    return Dashboard(tmp_path, mock_db_manager)


def test_dashboard_init(tmp_path, mock_db_manager):
    d = Dashboard(tmp_path, mock_db_manager)
    assert d.output_dir == tmp_path
    assert d.report_path == tmp_path / "dashboard.html"
    assert tmp_path.exists()


def test_update_creates_html(dashboard):
    data = DashboardData(
        generations=[0, 1], rmse_values=[0.5, 0.4], structure_counts=[100, 200], status="training"
    )

    dashboard.update(data)

    assert dashboard.report_path.exists()
    content = dashboard.report_path.read_text()
    assert "<html>" in content
    assert "Current Status: training" in content
    assert "Latest RMSE: 0.4" in content
    assert "data:image/png;base64" in content


def test_update_empty_data(dashboard):
    data = DashboardData()
    dashboard.update(data)

    assert dashboard.report_path.exists()
    content = dashboard.report_path.read_text()
    assert "No data yet" in content

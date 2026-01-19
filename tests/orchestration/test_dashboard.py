from pathlib import Path
from unittest.mock import MagicMock

from mlip_autopipec.core.database import DatabaseManager
from mlip_autopipec.orchestration.dashboard import Dashboard


def test_dashboard_init(tmp_path: Path) -> None:
    mock_db = MagicMock(spec=DatabaseManager)
    dashboard = Dashboard(tmp_path, mock_db)
    assert dashboard.output_dir == tmp_path
    assert dashboard.report_path == tmp_path / "dashboard.html"


def test_dashboard_update(tmp_path: Path) -> None:
    mock_db = MagicMock(spec=DatabaseManager)
    dashboard = Dashboard(tmp_path, mock_db)

    from mlip_autopipec.orchestration.models import DashboardData
    data = DashboardData(
        generations=[0, 1],
        rmse_values=[0.5, 0.1],
        structure_counts=[100, 200],
        status="training"
    )

    dashboard.update(data)

    assert dashboard.report_path.exists()
    assert "training" in dashboard.report_path.read_text()

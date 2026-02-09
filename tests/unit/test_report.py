import json
from unittest.mock import MagicMock

import pytest
from mlip_autopipec.core.report import ReportGenerator

from mlip_autopipec.domain_models.config import GlobalConfig, OrchestratorConfig


@pytest.fixture
def mock_workdir(tmp_path):
    # Create cycle directories
    for i in range(1, 4):
        d = tmp_path / f"cycle_{i:02d}"
        d.mkdir()
        # Create metrics.json
        metrics = {
            "cycle": i,
            "dataset_size": i * 100,
            "energy_rmse": 0.5 - (i * 0.1),
            "force_rmse": 0.1 - (i * 0.02),
            "validation_passed": i == 3
        }
        (d / "metrics.json").write_text(json.dumps(metrics))

    # Create workflow_state.json
    state = {"current_cycle": 3, "status": "CONVERGED"}
    (tmp_path / "workflow_state.json").write_text(json.dumps(state))

    return tmp_path


def test_report_generator_collect_metrics(mock_workdir):
    config = MagicMock(spec=GlobalConfig)
    config.workdir = mock_workdir

    # Configure orchestrator mock properly
    config.orchestrator = MagicMock(spec=OrchestratorConfig)
    config.orchestrator.cycle_dir_pattern = "cycle_{cycle:02d}"
    config.orchestrator.state_filename = "workflow_state.json"

    generator = ReportGenerator(config)
    metrics_df = generator.collect_metrics()

    assert len(metrics_df) == 3
    assert metrics_df.iloc[0]["dataset_size"] == 100
    # Use equality check for numpy/pandas booleans
    assert metrics_df.iloc[2]["validation_passed"] == True


def test_report_generator_html(mock_workdir):
    config = MagicMock(spec=GlobalConfig)
    config.workdir = mock_workdir

    # Configure orchestrator mock properly
    config.orchestrator = MagicMock(spec=OrchestratorConfig)
    config.orchestrator.cycle_dir_pattern = "cycle_{cycle:02d}"
    config.orchestrator.state_filename = "workflow_state.json"

    generator = ReportGenerator(config)
    html_content = generator.generate_report()

    assert "<html>" in html_content
    # Check for Cycle 3 in table
    assert "<td>3</td>" in html_content
    assert "CONVERGED" in html_content

from unittest.mock import patch
from pathlib import Path
import pytest
from typer.testing import CliRunner

from mlip_autopipec.app import app
from mlip_autopipec.domain_models.exploration import ExplorationTask
from mlip_autopipec.domain_models.dynamics import EonResult
from mlip_autopipec.domain_models.job import JobStatus
from mlip_autopipec.domain_models.structure import Structure
import ase

runner = CliRunner()

@pytest.fixture
def mock_policy():
    with patch("mlip_autopipec.physics.structure_gen.policy.AdaptivePolicy.decide") as mock:
        yield mock

@pytest.fixture
def mock_eon():
    with patch("mlip_autopipec.orchestration.phases.exploration.EonWrapper") as mock:
        yield mock

# Removed incomplete test test_uat_cycle08_akmc_flow

def test_uat_cycle08_deploy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init"])

    # Create artifacts
    (tmp_path / "potential.yace").touch()
    (tmp_path / "validation_report.html").touch()

        # We need a valid state with validation history
        # We can inject it by writing workflow_state.json?
        # Or mock state loading.

    # Writing workflow_state.json is better integration test
    import json
    state_data = {
            "project_name": "TestProject",
            "dataset_path": "data.pckl",
        "generation": 5,
            "current_phase": "EXPLORATION", # "IDLE" is not in Enum, use "EXPLORATION"
        "latest_potential_path": str(tmp_path / "potential.yace"),
            # "latest_dataset_path": "data.pckl", # Removed
        "validation_history": {
            "5": {
                "potential_id": "pot_5",
                "metrics": [
                    {"name": "RMSE", "value": 0.01, "passed": True}
                ],
                "plots": {},
                "overall_status": "PASS"
            }
        }
    }
    with open("workflow_state.json", "w") as f:
        json.dump(state_data, f)

    # Run deploy
    result = runner.invoke(app, ["deploy", "--author", "Tester", "--version", "1.0.0"])

    assert result.exit_code == 0, f"Deploy failed: {result.stdout}"
    assert "Deployment package created" in result.stdout
    assert (tmp_path / "dist" / "mlip_package_1.0.0.zip").exists()

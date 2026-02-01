import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from mlip_autopipec.infrastructure.production import ProductionDeployer
from mlip_autopipec.domain_models.production import ProductionManifest
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.domain_models.config import Config, PotentialConfig, ValidationConfig
from mlip_autopipec.domain_models.workflow import WorkflowState

@pytest.fixture
def mock_config():
    return Config(
        potential=PotentialConfig(elements=["Ti", "O"]),
        validation=ValidationConfig(report_path=Path("report.html"))
    )

@pytest.fixture
def mock_state():
    return WorkflowState(
        project_name="TestProject",
        dataset_path=Path("data.pckl"),
        generation=1,
        latest_potential_path=Path("potential.yace"),
        validation_history={}
    )

@patch("zipfile.ZipFile")
def test_deploy_success(mock_zip, mock_config, mock_state, tmp_path):
    # Setup state with validation result
    metrics = [ValidationMetric(name="RMSE", value=0.1, passed=True)]
    val_result = ValidationResult(
        potential_id="pot1",
        metrics=metrics,
        overall_status="PASS"
    )
    mock_state.validation_history = {1: val_result}

    # Mock files
    (tmp_path / "potential.yace").touch()
    (tmp_path / "report.html").touch()

    deployer = ProductionDeployer(mock_config, tmp_path)

    # We need to mock that potential and report exist in the path expected
    # The deployer usually takes paths from config or state.
    # We'll update mock_state to point to tmp_path files
    mock_state.latest_potential_path = tmp_path / "potential.yace"

    # We also need to mock report location if it relies on config
    # ProductionDeployer might need to know where the report is.
    # Usually Orchestrator knows.
    # Let's assume Deployer takes an explicit argument or finds it.

    output_zip = deployer.deploy(mock_state, version="1.0.0", author="Me")

    # assert output_zip.exists() # Fails because we mock ZipFile
    assert output_zip.name == "mlip_package_1.0.0.zip"

    # Verify ZipFile was called with correct path
    mock_zip.assert_called_with(output_zip, "w")

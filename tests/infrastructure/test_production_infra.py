import pytest
from unittest.mock import patch
from pathlib import Path
from mlip_autopipec.infrastructure.production import ProductionDeployer
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

def test_deploy_success(mock_config, mock_state, tmp_path):
    # Integration test without mocking ZipFile
    metrics = [ValidationMetric(name="RMSE", value=0.1, passed=True)]
    val_result = ValidationResult(
        potential_id="pot1",
        metrics=metrics,
        overall_status="PASS"
    )
    mock_state.validation_history = {1: val_result}

    # Create real dummy artifacts
    pot_path = tmp_path / "potential.yace"
    pot_path.write_text("dummy potential content")

    report_path = tmp_path / "report.html"
    report_path.write_text("<html>Report</html>")

    # Update config and state to point to real files
    mock_config.validation.report_path = report_path
    mock_state.latest_potential_path = pot_path

    # Use a subdirectory for output
    dist_dir = tmp_path / "dist"
    deployer = ProductionDeployer(mock_config, dist_dir)

    # Execute
    output_zip = deployer.deploy(mock_state, version="1.0.0", author="Me")

    # Verify
    assert output_zip.exists()
    assert output_zip.name == "mlip_package_1.0.0.zip"

    # Verify content of ZIP
    import zipfile
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        assert "potential.yace" in names
        assert "validation_report.html" in names
        assert "metadata.json" in names

        # Verify content integrity
        assert zf.read("potential.yace").decode() == "dummy potential content"

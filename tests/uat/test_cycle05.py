from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from mlip_autopipec.app import app
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric

runner = CliRunner()

def test_uat_c05_01_stable_potential(tmp_path):
    """
    Scenario 5.1: Validate a stable potential.
    Success Criteria:
    - Run validation on a known good potential.
    - Status is "PASS".
    - Report generated.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # 1. Setup
        runner.invoke(app, ["init"])
        potential_path = Path("potential.yace")
        potential_path.touch()

        # 2. Mock ValidationRunner.validate to return PASS results
        mock_results = [
            ValidationResult(
                potential_id="potential",
                metrics=[
                    ValidationMetric(name="Bulk Modulus", value=76.0, passed=True),
                    ValidationMetric(name="C11", value=100.0, passed=True),
                    ValidationMetric(name="Min Frequency", value=1.0, passed=True)
                ],
                plots={"eos": Path("eos.png")},
                overall_status="PASS"
            )
        ]

        with patch("mlip_autopipec.physics.validation.runner.ValidationRunner.validate", return_value=mock_results) as mock_validate:
             # Mock Structure generation to avoid needing ASE data/build issues if any (though unit tests passed)
             # But init config defaults to Si diamond, which is fine.

             # 3. Execute
             result = runner.invoke(app, ["validate", "--potential", "potential.yace"])

             # 4. Verify
             if result.exit_code != 0:
                 print(result.stdout)
             assert result.exit_code == 0
             assert "Validation Report generated" in result.stdout
             assert "Potential potential: PASS" in result.stdout
             assert "Bulk Modulus: 76.0000 (PASS)" in result.stdout

             assert Path("validation_report.html").exists()

def test_uat_c05_02_unstable_potential(tmp_path):
    """
    Scenario 5.2: Catching a Bad Potential.
    Success Criteria:
    - Status is "FAIL".
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        potential_path = Path("bad.yace")
        potential_path.touch()

        mock_results = [
            ValidationResult(
                potential_id="bad",
                metrics=[
                    ValidationMetric(name="Min Frequency", value=-5.0, passed=False, message="Imaginary frequencies")
                ],
                plots={},
                overall_status="FAIL"
            )
        ]

        with patch("mlip_autopipec.physics.validation.runner.ValidationRunner.validate", return_value=mock_results):
             result = runner.invoke(app, ["validate", "--potential", "bad.yace"])

             assert result.exit_code == 0 # Command succeeds even if validation fails (it reports failure)
             assert "Potential bad: FAIL" in result.stdout
             assert "Min Frequency: -5.0000 (FAIL)" in result.stdout
             assert Path("validation_report.html").exists()

def test_uat_c05_03_report_content(tmp_path):
    """
    Scenario 5.3: HTML Report Generation content check.
    """
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(app, ["init"])
        potential_path = Path("potential.yace")
        potential_path.touch()

        mock_results = [
             ValidationResult(
                potential_id="potential",
                metrics=[ValidationMetric(name="Test", value=1.0, passed=True)],
                plots={"test_plot": Path("plot.png").absolute()}, # Use absolute to test path handling
                overall_status="PASS"
            )
        ]

        # Create dummy plot
        Path("plot.png").touch()

        with patch("mlip_autopipec.physics.validation.runner.ValidationRunner.validate", return_value=mock_results):
             runner.invoke(app, ["validate", "-p", "potential.yace"])

             report = Path("validation_report.html")
             assert report.exists()
             content = report.read_text()
             assert "<html" in content
             assert "Test" in content
             assert "PASS" in content

"""Tests for report generator."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.report import ReportGenerator


@pytest.fixture
def validation_result() -> ValidationResult:
    """Create a mock validation result."""
    return ValidationResult(
        passed=True,
        metrics={"rmse_energy": 0.001, "bulk_modulus": 100.0},
        phonon_stable=True,
        elastic_stable=True,
        artifacts={"eos": "eos.png"},
    )


def test_generate_report(validation_result: ValidationResult, tmp_path: Path) -> None:
    """Test report generation."""
    output_path = tmp_path / "report.html"

    with patch("pyacemaker.validator.report.Environment") as MockEnv:
        mock_env = MockEnv.return_value
        mock_template = mock_env.get_template.return_value
        mock_template.render.return_value = "<html>Mock Report</html>"

        generator = ReportGenerator()
        generator.generate(validation_result, output_path)

        assert output_path.exists()
        assert output_path.read_text() == "<html>Mock Report</html>"

        # Verify template rendering context
        mock_template.render.assert_called_once()

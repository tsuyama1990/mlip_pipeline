"""Tests for ReportGenerator."""

from pathlib import Path

import pytest

from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.report import ReportGenerator


@pytest.fixture
def validation_result() -> ValidationResult:
    """Fixture for validation result."""
    return ValidationResult(
        passed=True,
        metrics={"bulk_modulus": 100.0},
        eos_stable=True,
        phonon_stable=True,
        elastic_stable=True,
        artifacts={"plot": "plot.png"},
    )


def test_report_generation(validation_result: ValidationResult, tmp_path: Path) -> None:
    """Test report generation."""
    # Ensure template dir is found relative to the source code
    # report.py uses Path(__file__).parent / "templates"

    generator = ReportGenerator()
    output_path = tmp_path / "report.html"

    generator.generate(validation_result, output_path)

    assert output_path.exists()
    content = output_path.read_text()

    # Check for key content
    assert "Validation Report" in content
    assert "PASSED" in content
    assert "bulk_modulus" in content
    assert "100.0" in content
    assert "plot.png" in content
    assert "EOS Stability" in content
    assert "Phonon Stability" in content
    assert "Elastic Stability" in content

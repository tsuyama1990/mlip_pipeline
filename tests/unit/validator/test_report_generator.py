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
        metrics={"bulk_modulus": 100.0, "C11": 200.0},
        eos_stable=True,
        phonon_stable=True,
        elastic_stable=True,
        artifacts={"plot": "plot.png"},
    )


def test_report_generation(validation_result: ValidationResult, tmp_path: Path) -> None:
    """Test report generation."""
    generator = ReportGenerator()
    output_path = tmp_path / "report.html"

    generator.generate(validation_result, output_path)

    assert output_path.exists()
    content = output_path.read_text()

    # Check for key content
    assert "Validation Report" in content
    assert "PASSED" in content
    # Check data binding
    assert "bulk_modulus" in content
    assert "100.0" in content
    assert "C11" in content
    assert "200.0" in content
    # Check artifacts
    assert "plot.png" in content
    # Check status flags
    assert "EOS Stability" in content
    assert "Phonon Stability" in content
    assert "Elastic Stability" in content

    # Basic HTML structure check
    assert "<html>" in content or "<!DOCTYPE html>" in content
    assert "</html>" in content
    assert "<body>" in content

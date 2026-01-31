import pytest
from pathlib import Path
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.physics.reporting.html_gen import ReportGenerator

def test_html_generator(tmp_path):
    results = [
        ValidationResult(
            potential_id="pot_001",
            metrics=[
                ValidationMetric(name="Bulk Modulus", value=70.0, passed=True),
                ValidationMetric(name="Phonon Stability", value=0.0, passed=False, message="Imaginary modes")
            ],
            plots={},
            overall_status="FAIL"
        )
    ]

    generator = ReportGenerator(output_dir=tmp_path)
    report_path = generator.generate_report(results)

    assert report_path.exists()
    assert report_path.name == "validation_report.html"

    content = report_path.read_text()
    assert "pot_001" in content
    assert "Bulk Modulus" in content
    assert "FAIL" in content
    assert "Imaginary modes" in content

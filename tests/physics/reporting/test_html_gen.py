from pathlib import Path

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.physics.reporting.html_gen import generate_report


def test_generate_report(tmp_path: Path) -> None:
    res = ValidationResult(
        potential_id="test_pot",
        metrics=[ValidationMetric(name="M1", value=1.0, passed=True)],
        plots={},
        overall_status="PASS",
    )

    report_path = tmp_path / "report.html"
    generate_report(res, report_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "test_pot" in content
    assert "PASS" in content
    assert "M1" in content

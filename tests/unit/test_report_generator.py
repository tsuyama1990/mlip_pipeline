

from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.validation.report_generator import ReportGenerator


def test_report_generator_success(tmp_path):
    generator = ReportGenerator()

    metric = MetricResult(
        name="test_metric",
        passed=True,
        score=0.99,
        details={"info": "good"},
        plot_path=tmp_path / "plot.png",
    )
    result = ValidationResult(passed=True, metrics=[metric])

    report_path = generator.generate(result, tmp_path)

    assert report_path.exists()
    content = report_path.read_text()
    assert "Validation Report" in content
    assert "PASSED" in content
    assert "test_metric" in content
    assert "plot.png" in content

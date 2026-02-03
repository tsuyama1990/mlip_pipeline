from pathlib import Path

from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.validation.report_generator import ReportGenerator


class TestReportGenerator:
    def test_generate_report(self, tmp_path: Path) -> None:
        """Test that HTML report is generated correctly."""
        # Create dummy result
        m1 = MetricResult(name="Test1", passed=True, score=1.0)
        m2 = MetricResult(name="Test2", passed=False, score=-1.0, details={"info": "bad"})
        val_result = ValidationResult(passed=False, metrics=[m1, m2])

        # Generate report
        report_path = tmp_path / "report.html"
        # Assuming ReportGenerator has a static method or class method
        ReportGenerator.generate(val_result, report_path)

        assert report_path.exists()
        content = report_path.read_text()
        assert "<html" in content
        assert "Test1" in content
        assert "PASS" in content  # From metric status
        assert "FAILED" in content # From global status

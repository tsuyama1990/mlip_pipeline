from pathlib import Path

import pytest

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.validation.report import ReportGenerator


@pytest.fixture
def dummy_results():
    return [
        ValidationResult(
            module="phonon",
            passed=True,
            metrics=[ValidationMetric(name="min_freq", value=0.5, passed=True)]
        ),
        ValidationResult(
            module="elastic",
            passed=False,
            metrics=[ValidationMetric(name="C11", value=150.0, passed=True)],
            error="Instability detected"
        )
    ]

class TestReportGenerator:
    def test_generate_report(self, dummy_results, tmp_path):
        work_dir = tmp_path / "report"
        work_dir.mkdir()

        generator = ReportGenerator(work_dir)
        report_path = generator.generate(dummy_results, Path("pot.yace"))

        assert report_path.exists()
        assert report_path.name == "validation_report.html"

        content = report_path.read_text()
        assert "<html>" in content
        assert "phonon" in content.lower()
        assert "elastic" in content.lower()
        assert "Instability detected" in content

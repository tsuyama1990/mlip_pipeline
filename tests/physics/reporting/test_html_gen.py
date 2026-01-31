from unittest.mock import patch
from pathlib import Path
from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric

def test_report_generation(tmp_path):
    # Depending on how imports are set up, this might fail if package not created.
    # But we will create it.
    from mlip_autopipec.physics.reporting.html_gen import ReportGenerator

    result = ValidationResult(
        potential_id="pot1",
        metrics=[ValidationMetric(name="M1", value=1.0, passed=True)],
        plots={"p1": Path("plot.png")},
        overall_status="PASS"
    )

    # We mock jinja2 Environment to return a dummy string
    with patch("mlip_autopipec.physics.reporting.html_gen.Environment") as MockEnv:
        mock_tmpl = MockEnv.return_value.get_template.return_value
        mock_tmpl.render.return_value = "<html>REPORT</html>"

        generator = ReportGenerator(output_dir=tmp_path)
        report_path = generator.generate(result)

        assert report_path.exists()
        assert report_path.read_text() == "<html>REPORT</html>"
        assert report_path.name == "validation_report.html"

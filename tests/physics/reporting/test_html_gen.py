from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult
from mlip_autopipec.physics.reporting.html_gen import ReportGenerator


def test_report_generation(tmp_path):
    """Test that HTML report is generated correctly."""
    # Create dummy plot
    plot_path = tmp_path / "plot.png"
    plot_path.write_bytes(b"fake png content")

    result = ValidationResult(
        potential_id="test.yace",
        metrics=[ValidationMetric(name="Test", value=1.0, passed=True)],
        plots={"Test Plot": plot_path},
        overall_status="PASS",
    )

    generator = ReportGenerator(tmp_path)
    report_path = generator.generate(result)

    assert report_path.exists()
    content = report_path.read_text()
    assert "Validation Report" in content
    assert "test.yace" in content
    assert "PASS" in content
    assert "data:image/png;base64," in content

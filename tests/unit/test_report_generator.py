from pathlib import Path

from mlip_autopipec.validator.elastic import ElasticResults
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults
from mlip_autopipec.validator.report import ReportGenerator


def test_report_generator(tmp_path: Path) -> None:
    """Test HTML report generation."""
    generator = ReportGenerator()

    eos_res = EOSResults(E0=-3.5, V0=11.0, B0=130.0, B0_prime=5.0)
    elastic_res = ElasticResults(C11=200.0, C12=100.0, C44=50.0, B=133.3, G=60.0)
    phonon_res = PhononResults(is_stable=True, max_imaginary_freq=0.0, band_structure_path=None)

    report_path = tmp_path / "report.html"

    generated_path = generator.generate_report(
        eos_results=eos_res,
        elastic_results=elastic_res,
        phonon_results=phonon_res,
        output_path=report_path
    )

    assert generated_path == report_path
    assert report_path.exists()

    content = report_path.read_text()
    assert "<html>" in content
    assert "B0: 130.0" in content
    assert "C11: 200.0" in content
    assert "Stable: True" in content
    assert "Max Imaginary Freq: 0.0" in content

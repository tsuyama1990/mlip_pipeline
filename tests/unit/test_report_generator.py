import numpy as np

from mlip_autopipec.validator.elastic import ElasticResults
from mlip_autopipec.validator.eos import EOSResults
from mlip_autopipec.validator.phonon import PhononResults
from mlip_autopipec.validator.report import ReportGenerator


def test_report_generator_full(tmp_path):
    generator = ReportGenerator()

    elastic = ElasticResults(C11=100.0, C12=50.0, C44=30.0, bulk_modulus=70.0, shear_modulus=30.0)
    eos = EOSResults(volume=15.0, energy=-5.0, bulk_modulus=70.0, bulk_modulus_derivative=4.0)

    # Mock phonon plot data
    phonon_plot_data = {
        "distances": [np.array([0.0, 0.5]), np.array([0.5, 1.0])],
        "frequencies": [np.array([[0.0, 1.0], [0.0, 1.5]]), np.array([[0.0, 1.5], [0.0, 2.0]])],
        "qpoints": [],
        "labels": []
    }

    phonon = PhononResults(max_imaginary_freq=0.0, is_stable=True, band_structure_plot_data=phonon_plot_data)

    report_path = generator.generate(
        output_dir=tmp_path,
        passed=True,
        elastic_results=elastic,
        eos_results=eos,
        phonon_results=phonon
    )

    assert report_path.exists()
    content = report_path.read_text()

    assert "PASSED" in content
    assert "C11" in content
    assert "100.00" in content
    assert "Equilibrium Volume" in content
    assert "15.000 A^3" in content
    assert "Phonon Stability" in content
    assert "Stable" in content
    assert "phonon_dispersion.png" in content

    assert (tmp_path / "phonon_dispersion.png").exists()

def test_report_generator_minimal(tmp_path):
    generator = ReportGenerator()

    report_path = generator.generate(
        output_dir=tmp_path,
        passed=False
    )

    assert report_path.exists()
    content = report_path.read_text()

    assert "FAILED" in content
    assert "No elastic analysis performed" in content
    assert "No EOS analysis performed" in content
    assert "No phonon analysis performed" in content

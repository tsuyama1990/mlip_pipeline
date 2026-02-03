from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult
from mlip_autopipec.validation.metrics import ElasticValidator, PhononValidator

try:
    from mlip_autopipec.validation.report_generator import ReportGenerator
except ImportError:
    ReportGenerator = MagicMock()


@pytest.fixture
def mock_atoms() -> Atoms:
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[3.6, 3.6, 3.6], pbc=True)
    atoms.calc = EMT() # Attach a calculator
    return atoms


class TestElasticValidator:
    def test_run_success(self, mock_atoms: Atoms) -> None:
        # We mock the internal helper that computes Cij to avoid calling real LAMMPS/ASE logic
        if isinstance(ElasticValidator, MagicMock):
            pytest.skip("ElasticValidator not implemented yet")

        with patch.object(
            ElasticValidator,
            "_compute_elastic_constants",
            return_value={"C11": 100, "C12": 50, "C44": 40},
        ):
            result = ElasticValidator.validate(Path("pot.yace"), mock_atoms)

        assert result.passed is True
        assert result.name == "Elastic Constants"
        assert result.score is not None
        assert result.details["C11"] == 100

    def test_run_failure(self, mock_atoms: Atoms) -> None:
        if isinstance(ElasticValidator, MagicMock):
            pytest.skip("ElasticValidator not implemented yet")

        # Unstable C11 < 0
        with patch.object(
            ElasticValidator,
            "_compute_elastic_constants",
            return_value={"C11": -100, "C12": 50, "C44": 40},
        ):
            result = ElasticValidator.validate(Path("pot.yace"), mock_atoms)

        assert result.passed is False


class TestPhononValidator:
    def test_run_stable(self, mock_atoms: Atoms) -> None:
        if isinstance(PhononValidator, MagicMock):
            pytest.skip("PhononValidator not implemented yet")

        # We patch Phonopy class.
        # Since phonopy is installed now, we can patch it.
        with (
            patch("mlip_autopipec.validation.metrics.Phonopy"),
            patch.object(PhononValidator, "_run_phonopy_checks", return_value=(True, 0.0, {})),
        ):
            result = PhononValidator.validate(Path("pot.yace"), mock_atoms)

        assert result.passed is True
        assert result.name == "Phonon Stability"


class TestReportGenerator:
    def test_generate_report(self, tmp_path: Path) -> None:
        if isinstance(ReportGenerator, MagicMock):
            pytest.skip("ReportGenerator not implemented yet")

        res = ValidationResult(
            passed=True, metrics=[MetricResult(name="Test", passed=True, score=1.0)]
        )

        with patch("mlip_autopipec.validation.report_generator.Environment") as mock_env:
            mock_tmpl = mock_env.return_value.get_template.return_value
            mock_tmpl.render.return_value = "<html>Report</html>"

            report_path = ReportGenerator.generate(res, tmp_path)

            assert report_path.exists()
            assert report_path.read_text() == "<html>Report</html>"

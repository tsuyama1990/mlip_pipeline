from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def validation_config() -> ValidationConfig:
    return ValidationConfig(
        run_validation=True,
        check_phonons=True,
        check_elastic=True,
        validation_structure="Cu.xyz",
    )


def test_validation_runner_disabled() -> None:
    config = ValidationConfig(run_validation=False)
    runner = ValidationRunner(config)
    result = runner.validate(Path("potential.yace"), Path("work_dir"))
    assert result.passed
    assert result.reason == "Validation disabled"


def test_validation_runner_success(validation_config: ValidationConfig, tmp_path: Path) -> None:
    runner = ValidationRunner(validation_config)

    # Mock structure
    mock_atoms = Atoms("Cu", positions=[[0, 0, 0]])

    # Mock dependencies
    with (
        patch("mlip_autopipec.validation.runner.read", return_value=mock_atoms),
        patch("mlip_autopipec.validation.runner.LAMMPS"),
        patch("mlip_autopipec.validation.metrics.PhononValidator.validate") as mock_phonon,
        patch("mlip_autopipec.validation.metrics.ElasticValidator.validate") as mock_elastic,
        patch("mlip_autopipec.validation.report_generator.ReportGenerator.generate") as mock_report,
    ):
        mock_phonon.return_value = MetricResult(name="Phonon", passed=True)
        mock_elastic.return_value = MetricResult(name="Elastic", passed=True)
        mock_report.return_value = tmp_path / "report.html"

        result = runner.validate(Path("potential.yace"), tmp_path)

        assert result.passed
        assert len(result.metrics) == 2
        assert result.report_path == tmp_path / "report.html"


def test_validation_runner_failure(validation_config: ValidationConfig, tmp_path: Path) -> None:
    runner = ValidationRunner(validation_config)

    mock_atoms = Atoms("Cu")

    with (
        patch("mlip_autopipec.validation.runner.read", return_value=mock_atoms),
        patch("mlip_autopipec.validation.runner.LAMMPS"),
        patch("mlip_autopipec.validation.metrics.PhononValidator.validate") as mock_phonon,
        patch("mlip_autopipec.validation.metrics.ElasticValidator.validate") as mock_elastic,
        patch("mlip_autopipec.validation.report_generator.ReportGenerator.generate"),
    ):
        mock_phonon.return_value = MetricResult(name="Phonon", passed=False)
        mock_elastic.return_value = MetricResult(name="Elastic", passed=True)

        result = runner.validate(Path("potential.yace"), tmp_path)

        assert not result.passed
        assert result.metrics[0].passed is False

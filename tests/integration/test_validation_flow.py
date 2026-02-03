from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def mock_config() -> ValidationConfig:
    return ValidationConfig(check_phonons=True, check_elastic=True)


def test_validation_runner_full_pass(mock_config: ValidationConfig, tmp_path: Path) -> None:
    if isinstance(ValidationRunner, MagicMock):
        pytest.skip("ValidationRunner not implemented")

    runner = ValidationRunner(mock_config)

    with (
        patch("mlip_autopipec.validation.runner.PhononValidator") as mock_pv,
        patch("mlip_autopipec.validation.runner.ElasticValidator") as mock_ev,
        patch("mlip_autopipec.validation.runner.ReportGenerator") as mock_rg,
    ):
        mock_pv.validate.return_value = MetricResult(name="Phonons", passed=True, score=0.0)
        mock_ev.validate.return_value = MetricResult(name="Elastic", passed=True, score=0.0)
        mock_rg.generate.return_value = tmp_path / "report.html"

        # Mock potential file
        pot_path = tmp_path / "potential.yace"
        pot_path.touch()

        # Run
        result = runner.validate(pot_path, tmp_path)

        assert result.passed is True
        assert len(result.metrics) == 2
        assert result.report_path is not None


def test_validation_runner_fail(mock_config: ValidationConfig, tmp_path: Path) -> None:
    if isinstance(ValidationRunner, MagicMock):
        pytest.skip("ValidationRunner not implemented")

    runner = ValidationRunner(mock_config)

    with (
        patch("mlip_autopipec.validation.runner.PhononValidator") as mock_pv,
        patch("mlip_autopipec.validation.runner.ElasticValidator") as mock_ev,
        patch("mlip_autopipec.validation.runner.ReportGenerator") as mock_rg,
    ):
        mock_pv.validate.return_value = MetricResult(name="Phonons", passed=False, score=1.0)
        mock_ev.validate.return_value = MetricResult(name="Elastic", passed=True, score=0.0)
        mock_rg.generate.return_value = tmp_path / "report.html"

        pot_path = tmp_path / "potential.yace"
        pot_path.touch()

        result = runner.validate(pot_path, tmp_path)

        assert result.passed is False
        assert len(result.metrics) == 2

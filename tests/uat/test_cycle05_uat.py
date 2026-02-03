from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult
from mlip_autopipec.validation.runner import ValidationRunner


def test_uat_report_generation(tmp_path: Path) -> None:
    """
    User Journey:
    1. User has a trained potential.
    2. Validation runs.
    3. User opens report.html.
    """
    if isinstance(ValidationRunner, MagicMock):
        pytest.skip("ValidationRunner not implemented")

    config = ValidationConfig(run_validation=True)
    runner = ValidationRunner(config)

    # Mock validators to pass
    with (
        patch("mlip_autopipec.validation.runner.PhononValidator") as mock_pv,
        patch("mlip_autopipec.validation.runner.ElasticValidator") as mock_ev,
        patch("mlip_autopipec.validation.runner.ReportGenerator") as mock_rg,
    ):
        mock_pv.validate.return_value = MetricResult(name="Phonons", passed=True, score=0.0)
        mock_ev.validate.return_value = MetricResult(name="Elastic", passed=True, score=0.0)

        report_file = tmp_path / "report.html"
        mock_rg.generate.return_value = report_file
        report_file.touch()  # Simulate creation

        pot_path = tmp_path / "pot.yace"
        pot_path.touch()

        result = runner.validate(pot_path, tmp_path)

        assert result.passed is True
        assert result.report_path == report_file
        assert report_file.exists()


def test_uat_gatekeeper_reject(tmp_path: Path) -> None:
    """
    User Journey:
    1. Potential is unstable.
    2. Validation fails.
    3. Status is REJECTED (implied by passed=False).
    """
    if isinstance(ValidationRunner, MagicMock):
        pytest.skip("ValidationRunner not implemented")

    config = ValidationConfig(run_validation=True)
    runner = ValidationRunner(config)

    with (
        patch("mlip_autopipec.validation.runner.PhononValidator") as mock_pv,
        patch("mlip_autopipec.validation.runner.ElasticValidator") as mock_ev,
        patch("mlip_autopipec.validation.runner.ReportGenerator") as mock_rg,
    ):
        mock_pv.validate.return_value = MetricResult(
            name="Phonons", passed=False, score=1.0
        )  # Fail
        mock_ev.validate.return_value = MetricResult(name="Elastic", passed=True, score=0.0)
        mock_rg.generate.return_value = tmp_path / "report.html"

        pot_path = tmp_path / "pot.yace"
        pot_path.touch()

        result = runner.validate(pot_path, tmp_path)

        assert result.passed is False

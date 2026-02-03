from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.validation.runner import ValidationRunner


class TestValidationFlow:
    @pytest.fixture
    def mock_config(self) -> ValidationConfig:
        return ValidationConfig(
            run_validation=True,
            check_phonons=True,
            check_elastic=True
        )

    def test_runner_execution(self, mock_config: ValidationConfig, tmp_path: Path) -> None:
        """Test that runner executes all validators and generates report."""
        runner = ValidationRunner(config=mock_config)

        potential_path = tmp_path / "potential.yace"
        potential_path.touch()
        work_dir = tmp_path / "validation"
        work_dir.mkdir()

        # Mock individual validators to avoid heavy computation
        with patch("mlip_autopipec.validation.runner.PhononValidator.run") as mock_phonon, \
             patch("mlip_autopipec.validation.runner.ElasticValidator.run") as mock_elastic, \
             patch("mlip_autopipec.validation.runner.ReportGenerator.generate") as mock_report:

            # Mock returns
            from mlip_autopipec.domain_models.validation import MetricResult
            mock_phonon.return_value = MetricResult(name="Phonon", passed=True)
            mock_elastic.return_value = MetricResult(name="Elastic", passed=True)

            # Mock structure loading if necessary, or ensure runner handles it
            # We assume runner generates/loads a structure. We might need to mock that too.
            with patch("mlip_autopipec.validation.runner.ValidationRunner._get_test_structure") as mock_struct:
                mock_struct.return_value = Atoms("Cu")

                result = runner.validate(potential_path, work_dir)

                assert isinstance(result, ValidationResult)
                assert result.passed is True
                assert len(result.metrics) == 2

                mock_phonon.assert_called_once()
                mock_elastic.assert_called_once()
                mock_report.assert_called_once()

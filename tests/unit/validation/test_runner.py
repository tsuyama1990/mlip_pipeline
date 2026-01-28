from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.validation import ValidationConfig
from mlip_autopipec.domain_models.validation import ValidationResult
from mlip_autopipec.validation.runner import ValidationRunner


@pytest.fixture
def validation_config():
    return ValidationConfig()

@pytest.fixture
def dummy_atoms():
    return Atoms("Si2")

class TestValidationRunner:
    @patch("mlip_autopipec.validation.runner.PhononValidator")
    @patch("mlip_autopipec.validation.runner.ElasticityValidator")
    @patch("mlip_autopipec.validation.runner.EOSValidator")
    @patch("mlip_autopipec.validation.runner.ReportGenerator")
    def test_run_all(self, mock_report, mock_eos, mock_elastic, mock_phonon, validation_config, tmp_path, dummy_atoms):
        runner = ValidationRunner(validation_config, tmp_path)

        # Mock validators to return a dummy result
        dummy_result = ValidationResult(module="test", passed=True)

        mock_phonon.return_value.validate.return_value = dummy_result
        mock_elastic.return_value.validate.return_value = dummy_result
        mock_eos.return_value.validate.return_value = dummy_result

        results = runner.run(dummy_atoms, Path("pot.yace"))

        assert len(results) == 3
        mock_phonon.return_value.validate.assert_called_once()
        mock_elastic.return_value.validate.assert_called_once()
        mock_eos.return_value.validate.assert_called_once()

        # Verify report generation called
        mock_report.return_value.generate.assert_called_once_with(results, Path("pot.yace"))

    @patch("mlip_autopipec.validation.runner.PhononValidator")
    @patch("mlip_autopipec.validation.runner.ElasticityValidator")
    @patch("mlip_autopipec.validation.runner.EOSValidator")
    @patch("mlip_autopipec.validation.runner.ReportGenerator")
    def test_run_subset(self, mock_report, mock_eos, mock_elastic, mock_phonon, validation_config, tmp_path, dummy_atoms):
        runner = ValidationRunner(validation_config, tmp_path)

        dummy_result = ValidationResult(module="test", passed=True)
        mock_phonon.return_value.validate.return_value = dummy_result

        results = runner.run(dummy_atoms, Path("pot.yace"), modules=["phonon"])

        assert len(results) == 1
        mock_phonon.return_value.validate.assert_called_once()
        mock_elastic.return_value.validate.assert_not_called()
        mock_eos.return_value.validate.assert_not_called()

    def test_invalid_input(self, validation_config, tmp_path):
        runner = ValidationRunner(validation_config, tmp_path)
        with pytest.raises(TypeError):
            runner.run("not atoms", Path("pot.yace"))

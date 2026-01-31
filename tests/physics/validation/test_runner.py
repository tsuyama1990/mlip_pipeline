from unittest.mock import patch
from pathlib import Path

from mlip_autopipec.domain_models.validation import ValidationResult, ValidationMetric
from mlip_autopipec.domain_models.config import ValidationConfig, PotentialConfig


def test_runner_structure():
    """Simple test to ensure class structure exists."""
    from mlip_autopipec.physics.validation.runner import ValidationRunner, BaseValidator
    assert isinstance(ValidationRunner, type)
    assert isinstance(BaseValidator, type)


def test_runner_validate_flow():
    # Patch the classes where they are defined, since runner imports them inside method
    with patch("mlip_autopipec.physics.validation.eos.EOSValidator") as MockEOS, \
         patch("mlip_autopipec.physics.validation.elasticity.ElasticityValidator") as MockElast, \
         patch("mlip_autopipec.physics.validation.phonon.PhononValidator") as MockPhonon, \
         patch("mlip_autopipec.physics.validation.runner.ReportGenerator") as MockReport:

        from mlip_autopipec.physics.validation.runner import ValidationRunner

        # Setup mocks
        mock_eos = MockEOS.return_value
        mock_eos.validate.return_value = ValidationResult(
            potential_id="p", metrics=[ValidationMetric(name="EOS", value=1.0, passed=True)],
            plots={}, overall_status="PASS"
        )

        mock_elast = MockElast.return_value
        mock_elast.validate.return_value = ValidationResult(
            potential_id="p", metrics=[ValidationMetric(name="Elast", value=1.0, passed=True)],
            plots={}, overall_status="PASS"
        )

        mock_phonon = MockPhonon.return_value
        mock_phonon.validate.return_value = ValidationResult(
            potential_id="p", metrics=[ValidationMetric(name="Phonon", value=1.0, passed=True)],
            plots={}, overall_status="PASS"
        )

        mock_report_inst = MockReport.return_value

        runner = ValidationRunner(ValidationConfig(), PotentialConfig(elements=["Si"], cutoff=5.0))
        result = runner.validate(Path("potential.yace"))

        assert result.overall_status == "PASS"
        assert len(result.metrics) == 3

        # Verify calls
        mock_eos.validate.assert_called_once()
        mock_elast.validate.assert_called_once()
        mock_phonon.validate.assert_called_once()
        mock_report_inst.generate.assert_called_once()


def test_runner_aggregation_fail():
     with patch("mlip_autopipec.physics.validation.eos.EOSValidator") as MockEOS, \
         patch("mlip_autopipec.physics.validation.elasticity.ElasticityValidator") as MockElast, \
         patch("mlip_autopipec.physics.validation.phonon.PhononValidator") as MockPhonon, \
         patch("mlip_autopipec.physics.validation.runner.ReportGenerator"):

        from mlip_autopipec.physics.validation.runner import ValidationRunner

        # Setup mocks
        mock_eos = MockEOS.return_value
        mock_eos.validate.return_value = ValidationResult(
            potential_id="p", metrics=[ValidationMetric(name="EOS", value=1.0, passed=True)],
            plots={}, overall_status="PASS"
        )

        mock_elast = MockElast.return_value
        mock_elast.validate.return_value = ValidationResult(
            potential_id="p", metrics=[ValidationMetric(name="Elast", value=1.0, passed=False)],
            plots={}, overall_status="FAIL"
        )

        MockPhonon.return_value.validate.return_value = ValidationResult(
             potential_id="p", metrics=[], plots={}, overall_status="PASS"
        )

        runner = ValidationRunner(ValidationConfig(), PotentialConfig(elements=["Si"], cutoff=5.0))
        result = runner.validate(Path("potential.yace"))

        assert result.overall_status == "FAIL"

"""
UAT Script for Cycle 05: The Validator
Scenario: Verify correct validation pipeline execution.
"""
from pathlib import Path
from unittest.mock import patch

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult
from mlip_autopipec.validation.runner import ValidationRunner


def test_uat_scenario_01_report_card(tmp_path: Path) -> None:
    """
    Scenario 05-01: The Report Card
    User Journey: Check report generation after validation.
    """
    # Setup
    config = ValidationConfig(check_phonons=True, check_elastic=True)
    runner = ValidationRunner(config)
    work_dir = tmp_path / "active_learning/iter_005/validation"
    work_dir.mkdir(parents=True)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    # Execution (Mocked internals for UAT speed, focusing on flow)
    with patch("mlip_autopipec.validation.runner.PhononValidator.run") as m_phon, \
         patch("mlip_autopipec.validation.runner.ElasticValidator.run") as m_elas:

        m_phon.return_value = MetricResult(name="Phonon Stability", passed=True, score=0.0)
        m_elas.return_value = MetricResult(name="Elastic Stability", passed=True, details={"C11": 100})

        # We allow ReportGenerator to run effectively to verify the HTML creation
        # assuming we provide dummy data that works with the template.

        # We need to mock structure generation
        with patch("mlip_autopipec.validation.runner.ValidationRunner._get_test_structure") as m_struct:
            from ase import Atoms
            m_struct.return_value = Atoms("Cu")

            result = runner.validate(potential_path, work_dir)

    # Verification
    assert result.passed
    report_file = work_dir / "report.html"
    assert report_file.exists()
    content = report_file.read_text()
    assert "Phonon Stability" in content
    assert "Elastic Stability" in content
    assert "PASSED" in content


def test_uat_scenario_02_gatekeeper(tmp_path: Path) -> None:
    """
    Scenario 05-02: The Gatekeeper
    User Journey: Validator rejects unstable potential.
    """
    # Setup
    config = ValidationConfig()
    runner = ValidationRunner(config)
    work_dir = tmp_path / "validation_fail"
    work_dir.mkdir()
    potential_path = tmp_path / "bad.yace"
    potential_path.touch()

    with patch("mlip_autopipec.validation.runner.PhononValidator.run") as m_phon, \
         patch("mlip_autopipec.validation.runner.ElasticValidator.run") as m_elas, \
         patch("mlip_autopipec.validation.runner.ValidationRunner._get_test_structure") as m_struct:

        from ase import Atoms
        m_struct.return_value = Atoms("Cu")

        # Phonon fails
        m_phon.return_value = MetricResult(name="Phonon Stability", passed=False, score=-2.0)
        m_elas.return_value = MetricResult(name="Elastic Stability", passed=True)

        result = runner.validate(potential_path, work_dir)

    assert not result.passed
    assert result.reason is not None
    assert "Phonon" in result.reason or "failed" in result.reason.lower()

    print("Scenario 05-02 Passed: Unstable potential rejected.")

from pathlib import Path

import pytest
from mlip_autopipec.infrastructure.mocks import MockValidator
from mlip_autopipec.domain_models.validation import ValidationResult

def test_mock_validator(tmp_path: Path) -> None:
    # Setup
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    # Init
    validator = MockValidator()

    # Run
    result = validator.validate(potential_path)

    # Check
    assert isinstance(result, ValidationResult)
    assert result.passed is True
    assert result.metrics["rmse_energy"] == 0.001
    assert result.report_path == tmp_path / "validation_report.html"

def test_mock_validator_fail() -> None:
    validator = MockValidator(params={"passed": False})
    result = validator.validate(Path("dummy.yace"))
    assert result.passed is False

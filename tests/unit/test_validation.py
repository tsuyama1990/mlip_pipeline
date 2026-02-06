import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationResult


def test_validation_result() -> None:
    res = ValidationResult(metrics={"rmse": 0.1}, is_stable=True)
    assert res.metrics["rmse"] == 0.1
    assert res.is_stable is True

def test_validation_result_empty_metrics() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ValidationResult(metrics={}, is_stable=True)
    assert "Metrics dictionary cannot be empty" in str(excinfo.value)

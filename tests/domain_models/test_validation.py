import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


def test_validation_metric_valid():
    m = ValidationMetric(name="Test", value=1.0, passed=True)
    assert m.value == 1.0
    assert m.passed is True


def test_validation_metric_infinite_value():
    with pytest.raises(ValidationError):
        ValidationMetric(name="Test", value=float("inf"), passed=False)
    with pytest.raises(ValidationError):
        ValidationMetric(name="Test", value=float("nan"), passed=False)


def test_validation_result_valid():
    m = ValidationMetric(name="Test", value=1.0, passed=True)
    r = ValidationResult(potential_id="pot1", metrics=[m], overall_status="PASS")
    assert r.overall_status == "PASS"
    assert len(r.metrics) == 1

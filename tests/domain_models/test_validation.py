import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


def test_validation_metric_valid():
    metric = ValidationMetric(
        name="Test Metric",
        value=1.23,
        reference=1.20,
        error=0.03,
        passed=True,
    )
    assert metric.name == "Test Metric"
    assert metric.value == 1.23
    assert metric.reference == 1.20
    assert metric.passed is True


def test_validation_metric_minimal():
    metric = ValidationMetric(name="Test Metric", value=1.0, passed=False)
    assert metric.reference is None
    assert metric.error is None


def test_validation_result_valid():
    metric = ValidationMetric(name="Test", value=1.0, passed=True)
    result = ValidationResult(
        potential_id="pot_123",
        metrics=[metric],
        overall_status="PASS",
    )
    assert result.potential_id == "pot_123"
    assert len(result.metrics) == 1
    assert result.overall_status == "PASS"


def test_validation_result_default():
    result = ValidationResult(potential_id="pot_123")
    assert result.metrics == []
    assert result.plots == {}
    assert result.overall_status == "FAIL"  # Default should be conservative


def test_validation_result_invalid_status():
    with pytest.raises(ValidationError):
        ValidationResult(potential_id="pot_123", overall_status="INVALID")

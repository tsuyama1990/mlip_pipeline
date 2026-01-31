from pathlib import Path
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


def test_validation_metric_valid():
    metric = ValidationMetric(
        name="Test Metric",
        value=1.0,
        reference=1.2,
        error=0.2,
        passed=True
    )
    assert metric.name == "Test Metric"
    assert metric.value == 1.0
    assert metric.reference == 1.2
    assert metric.error == 0.2
    assert metric.passed is True


def test_validation_metric_minimal():
    metric = ValidationMetric(
        name="Test Metric",
        value=1.0,
        passed=False
    )
    assert metric.name == "Test Metric"
    assert metric.reference is None
    assert metric.passed is False


def test_validation_metric_invalid():
    with pytest.raises(ValidationError):
        ValidationMetric(name="Test", value="not a float", passed=True)


def test_validation_result_valid():
    metric = ValidationMetric(name="Test", value=1.0, passed=True)
    result = ValidationResult(
        potential_id="pot_001",
        metrics=[metric],
        plots={"test_plot": Path("/tmp/plot.png")},
        overall_status="PASS"
    )
    assert result.potential_id == "pot_001"
    assert len(result.metrics) == 1
    assert result.plots["test_plot"] == Path("/tmp/plot.png")
    assert result.overall_status == "PASS"


def test_validation_result_invalid_status():
    metric = ValidationMetric(name="Test", value=1.0, passed=True)
    with pytest.raises(ValidationError):
        ValidationResult(
            potential_id="pot_001",
            metrics=[metric],
            plots={},
            overall_status="INVALID"  # Should be PASS, WARN, FAIL
        )

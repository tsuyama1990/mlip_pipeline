from pathlib import Path
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


def test_validation_metric_valid():
    metric = ValidationMetric(
        name="Bulk Modulus",
        value=100.0,
        reference=98.0,
        error=2.0,
        passed=True
    )
    assert metric.name == "Bulk Modulus"
    assert metric.value == 100.0
    assert metric.reference == 98.0
    assert metric.passed is True


def test_validation_metric_invalid_extra():
    with pytest.raises(ValidationError):
        ValidationMetric(
            name="Test",
            value=1.0,
            passed=True,
            extra_field="fail"  # type: ignore
        )


def test_validation_result_valid():
    metric = ValidationMetric(name="M1", value=1.0, passed=True)
    result = ValidationResult(
        potential_id="pot1",
        metrics=[metric],
        plots={"p1": Path("plot.png")},
        overall_status="PASS"
    )
    assert result.potential_id == "pot1"
    assert result.overall_status == "PASS"
    assert len(result.metrics) == 1
    assert result.plots["p1"] == Path("plot.png")


def test_validation_result_invalid_status():
    metric = ValidationMetric(name="M1", value=1.0, passed=True)
    with pytest.raises(ValidationError):
        ValidationResult(
            potential_id="pot1",
            metrics=[metric],
            plots={},
            overall_status="UNKNOWN"  # type: ignore
        )

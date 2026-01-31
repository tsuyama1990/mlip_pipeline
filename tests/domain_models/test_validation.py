from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import ValidationMetric, ValidationResult


def test_validation_metric_valid():
    m = ValidationMetric(
        name="Bulk Modulus", value=100.0, reference=98.0, error=2.0, passed=True
    )
    assert m.name == "Bulk Modulus"
    assert m.value == 100.0
    assert m.passed is True


def test_validation_metric_minimal():
    m = ValidationMetric(name="Instability", value=0.0, passed=False)
    assert m.reference is None


def test_validation_result_valid():
    m = ValidationMetric(name="Test", value=1.0, passed=True)
    r = ValidationResult(
        potential_id="test_pot",
        metrics=[m],
        plots={"phonon": Path("phonon.png")},
        overall_status="PASS",
    )
    assert r.overall_status == "PASS"
    assert r.plots["phonon"] == Path("phonon.png")


def test_validation_result_invalid_status():
    m = ValidationMetric(name="Test", value=1.0, passed=True)
    with pytest.raises(ValidationError):
        ValidationResult(
            potential_id="test_pot",
            metrics=[m],
            plots={},
            overall_status="INVALID",  # type: ignore
        )

from pathlib import Path
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.validation import (
    ValidationConfig,
    ValidationMetric,
    ValidationResult,
)


def test_validation_config_defaults() -> None:
    config = ValidationConfig()
    assert config.phonon_tolerance == -0.1
    assert config.eos_vol_range == 0.1
    assert config.eos_n_points == 10
    assert config.elastic_strain_mag == 0.01


def test_validation_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(extra_field=1)  # type: ignore[call-arg]


def test_validation_metric() -> None:
    metric = ValidationMetric(
        name="Bulk Modulus",
        value=100.0,
        reference=98.0,
        error=2.0,
        passed=True,
        unit="GPa",
    )
    assert metric.name == "Bulk Modulus"
    assert metric.value == 100.0


def test_validation_result() -> None:
    metric = ValidationMetric(name="M1", value=1.0, passed=True)
    res = ValidationResult(
        potential_id="pot1",
        metrics=[metric],
        plots={"p1": Path("p1.png")},
        overall_status="PASS",
    )
    assert res.overall_status == "PASS"
    assert res.metrics[0].name == "M1"


def test_validation_result_status_literal() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(
            potential_id="pot1",
            metrics=[],
            plots={},
            overall_status="GOOD",  # type: ignore[arg-type]
        )

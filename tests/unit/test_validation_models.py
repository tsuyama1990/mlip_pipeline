
import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult


def test_validation_config_defaults() -> None:
    config = ValidationConfig()
    assert config.run_validation is True
    assert config.check_phonons is True
    assert config.check_elastic is True


def test_validation_config_custom() -> None:
    config = ValidationConfig(
        run_validation=False, check_phonons=False, check_elastic=False
    )
    assert config.run_validation is False
    assert config.check_phonons is False
    assert config.check_elastic is False


def test_metric_result_valid() -> None:
    metric = MetricResult(
        name="phonons", passed=True, score=0.0, details={"max_imag_freq": 0.0}
    )
    assert metric.name == "phonons"
    assert metric.passed is True
    assert metric.score == 0.0


def test_metric_result_minimal() -> None:
    metric = MetricResult(name="elastic", passed=False)
    assert metric.name == "elastic"
    assert metric.passed is False
    assert metric.score is None


def test_validation_result_valid() -> None:
    metric = MetricResult(name="phonons", passed=True)
    result = ValidationResult(passed=True, metrics=[metric])
    assert result.passed is True
    assert len(result.metrics) == 1
    assert result.metrics[0].name == "phonons"


def test_validation_result_invalid() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(passed=True, metrics={"invalid": "type"})  # type: ignore

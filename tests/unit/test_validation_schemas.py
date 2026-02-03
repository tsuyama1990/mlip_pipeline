import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import ValidationConfig
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult


def test_validation_config_defaults() -> None:
    config = ValidationConfig()
    assert config.run_validation is True
    assert config.check_phonons is True
    assert config.check_elastic is True
    assert config.validation_structure is None


def test_validation_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ValidationConfig(extra_field="fail")  # type: ignore[call-arg]


def test_metric_result_valid() -> None:
    metric = MetricResult(name="test", passed=True, score=0.1)
    assert metric.name == "test"
    assert metric.passed is True
    assert metric.score == 0.1
    assert metric.details == {}


def test_metric_result_optional_score() -> None:
    metric = MetricResult(name="test", passed=False)
    assert metric.score is None


def test_validation_result_valid() -> None:
    m1 = MetricResult(name="m1", passed=True)
    res = ValidationResult(passed=True, metrics=[m1])
    assert len(res.metrics) == 1
    assert res.metrics[0].name == "m1"


def test_validation_result_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ValidationResult(passed=True, unknown="fail")  # type: ignore[call-arg]

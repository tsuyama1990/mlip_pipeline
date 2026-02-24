"""Tests for ValidationResult."""

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.validator import ValidationResult


def test_validation_result_valid() -> None:
    """Test valid ValidationResult creation."""
    result = ValidationResult(
        passed=True,
        metrics={"bulk_modulus": 100.0},
        eos_stable=True,
        phonon_stable=True,
        elastic_stable=True,
        artifacts={"plot": "plot.png"},
    )
    assert result.passed is True
    assert result.metrics["bulk_modulus"] == 100.0
    assert result.eos_stable is True
    assert result.phonon_stable is True
    assert result.elastic_stable is True
    assert result.artifacts["plot"] == "plot.png"


def test_validation_result_invalid_extra() -> None:
    """Test ValidationResult rejects extra fields."""
    with pytest.raises(ValidationError):
        ValidationResult(
            passed=True,
            metrics={},
            eos_stable=True,
            phonon_stable=True,
            elastic_stable=True,
            extra_field="fail",
        )


def test_validation_result_missing() -> None:
    """Test ValidationResult rejects missing fields."""
    with pytest.raises(ValidationError):
        ValidationResult(
            passed=True,
            metrics={},
            # missing stability fields
        )

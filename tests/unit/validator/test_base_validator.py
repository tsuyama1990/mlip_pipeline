"""Tests for BaseValidator."""

from pathlib import Path

import pytest
from ase import Atoms

from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.base import BaseValidator


def test_base_validator_is_abstract() -> None:
    """Test BaseValidator cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseValidator()  # type: ignore[abstract]


def test_concrete_validator_implementation() -> None:
    """Test concrete implementation."""

    class MockValidator(BaseValidator):
        def validate(
            self,
            potential_path: Path,
            structure: Atoms,
            output_dir: Path,
            **kwargs: object,
        ) -> ValidationResult:
            return ValidationResult(
                passed=True,
                metrics={},
                eos_stable=True,
                phonon_stable=True,
                elastic_stable=True,
            )

    validator = MockValidator()
    result = validator.validate(Path("pot"), Atoms(), Path("out"))
    assert result.passed is True

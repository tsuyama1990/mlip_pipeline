"""Tests for Validator manager."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import ValidatorConfig
from pyacemaker.domain_models.models import Potential, PotentialType
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.manager import ValidatorManager


@pytest.fixture
def mock_atoms() -> Atoms:
    """Create a mock Atoms object."""
    return Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


@pytest.fixture
def validator_config() -> ValidatorConfig:
    """Create a validator configuration."""
    return ValidatorConfig()


def test_validator_run(
    mock_atoms: Atoms, validator_config: ValidatorConfig, tmp_path: Path
) -> None:
    """Test validator execution."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    potential = Potential(
        path=potential_path,
        type=PotentialType.PACE,
        version="1.0",
        metrics={},
        parameters={}
    )

    with (
        patch("pyacemaker.validator.manager.PhysicsValidator") as MockPhysicsValidator,
        patch("pyacemaker.validator.manager.ReportGenerator") as MockReportGenerator,
        patch("pyacemaker.validator.manager.validate_safe_path"),
    ):
        mock_validator = MockPhysicsValidator.return_value
        mock_validator.check_phonons.return_value = True
        mock_validator.check_eos.return_value = (100.0, "eos.png")
        mock_validator.check_elastic.return_value = (True, {"C11": 100.0})

        # We also need to mock _attach_calculator or ensure it works.
        # _attach_calculator tries to import mace or lammpslib.
        # We can patch _attach_calculator on the instance or class.
        with patch.object(ValidatorManager, "_attach_calculator", return_value=mock_atoms):
             manager = ValidatorManager(validator_config)
             result = manager.validate(potential, mock_atoms, output_dir=tmp_path)

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.phonon_stable is True
        assert result.elastic_stable is True
        assert result.eos_stable is True
        assert any(str(v).endswith("eos.png") for v in result.artifacts.values())

        MockReportGenerator.return_value.generate.assert_called_once()


def test_validator_failure(
    mock_atoms: Atoms, validator_config: ValidatorConfig, tmp_path: Path
) -> None:
    """Test validator failure."""
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    potential = Potential(
        path=potential_path,
        type=PotentialType.PACE,
        version="1.0",
        metrics={},
        parameters={}
    )

    with (
        patch("pyacemaker.validator.manager.PhysicsValidator") as MockPhysicsValidator,
        patch("pyacemaker.validator.manager.ReportGenerator"),
        patch("pyacemaker.validator.manager.validate_safe_path"),
    ):
        mock_validator = MockPhysicsValidator.return_value
        mock_validator.check_phonons.return_value = False  # Fail
        mock_validator.check_eos.return_value = (100.0, "eos.png")
        mock_validator.check_elastic.return_value = (True, {"C11": 100.0})

        with patch.object(ValidatorManager, "_attach_calculator", return_value=mock_atoms):
            manager = ValidatorManager(validator_config)
            result = manager.validate(potential, mock_atoms, output_dir=tmp_path)

        assert isinstance(result, ValidationResult)
        assert result.passed is False
        assert result.phonon_stable is False
        assert result.metrics["bulk_modulus"] == 100.0

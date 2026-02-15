"""Tests for Validator manager."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import ValidatorConfig
from pyacemaker.domain_models.validator import ValidationResult
from pyacemaker.validator.manager import ValidatorManager


@pytest.fixture
def mock_atoms():
    """Create a mock Atoms object."""
    return Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


@pytest.fixture
def validator_config():
    """Create a validator configuration."""
    return ValidatorConfig()


def test_validator_run(mock_atoms, validator_config, tmp_path):
    """Test validator execution."""
    manager = ValidatorManager(validator_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    with (
        patch("pyacemaker.validator.manager.check_phonons") as mock_phonons,
        patch("pyacemaker.validator.manager.check_eos") as mock_eos,
        patch("pyacemaker.validator.manager.check_elastic") as mock_elastic,
        patch("pyacemaker.validator.manager.ReportGenerator") as mock_report,
    ):
        mock_phonons.return_value = True
        mock_eos.return_value = (100.0, "eos.png")
        mock_elastic.return_value = (True, {"C11": 100.0})

        result = manager.validate(potential_path, mock_atoms, output_dir=tmp_path)

        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.phonon_stable is True
        assert result.elastic_stable is True
        assert any(str(v).endswith("eos.png") for v in result.artifacts.values())

        mock_report.return_value.generate.assert_called_once()


def test_validator_failure(mock_atoms, validator_config, tmp_path):
    """Test validator failure."""
    manager = ValidatorManager(validator_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    with (
        patch("pyacemaker.validator.manager.check_phonons") as mock_phonons,
        patch("pyacemaker.validator.manager.check_eos") as mock_eos,
        patch("pyacemaker.validator.manager.check_elastic") as mock_elastic,
        patch("pyacemaker.validator.manager.ReportGenerator"),
    ):
        mock_phonons.return_value = False  # Fail
        mock_eos.return_value = (100.0, "eos.png")
        mock_elastic.return_value = (True, {"C11": 100.0})

        result = manager.validate(potential_path, mock_atoms, output_dir=tmp_path)

        assert isinstance(result, ValidationResult)
        assert result.passed is False
        assert result.phonon_stable is False
        assert result.metrics["bulk_modulus"] == 100.0

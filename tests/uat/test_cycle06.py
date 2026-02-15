"""UAT Tests for Cycle 06."""

from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from pyacemaker.core.config import EONConfig, ValidatorConfig
from pyacemaker.dynamics.kmc import EONWrapper
from pyacemaker.validator.manager import ValidatorManager


@pytest.fixture
def mock_atoms():
    """Create a mock Atoms object."""
    return Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


@pytest.fixture
def validator_config():
    """Create a validator configuration."""
    return ValidatorConfig()


@pytest.fixture
def eon_config():
    """Create an EON configuration."""
    return EONConfig(executable="mock_eon")


def test_scenario_01_phonon_stability(mock_atoms, validator_config, tmp_path):
    """Scenario 01: Verify phonon stability check."""
    manager = ValidatorManager(validator_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    # Mock specific phonon check
    with (
        patch("pyacemaker.validator.manager.check_phonons") as mock_check,
        patch("pyacemaker.validator.manager.check_eos") as mock_eos,
        patch("pyacemaker.validator.manager.check_elastic") as mock_elastic,
        patch("pyacemaker.validator.manager.ReportGenerator"),
    ):
        mock_check.return_value = False  # Unstable
        mock_eos.return_value = (100.0, "eos.png")
        mock_elastic.return_value = (True, {})

        result = manager.validate(potential_path, mock_atoms, output_dir=tmp_path)

        assert result.phonon_stable is False
        assert result.passed is False


def test_scenario_02_eos_check(mock_atoms, validator_config, tmp_path):
    """Scenario 02: Verify EOS check."""
    manager = ValidatorManager(validator_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    with (
        patch("pyacemaker.validator.manager.check_eos") as mock_eos,
        patch("pyacemaker.validator.manager.check_phonons") as mock_phonons,
        patch("pyacemaker.validator.manager.check_elastic") as mock_elastic,
        patch("pyacemaker.validator.manager.ReportGenerator"),
    ):
        mock_eos.return_value = (150.0, "eos.png")
        mock_phonons.return_value = True
        mock_elastic.return_value = (True, {})

        result = manager.validate(potential_path, mock_atoms, output_dir=tmp_path)

        assert result.metrics["bulk_modulus"] == 150.0
        assert any(str(v).endswith("eos.png") for v in result.artifacts.values())


def test_scenario_03_report_generation(mock_atoms, validator_config, tmp_path):
    """Scenario 03: Verify HTML report generation."""
    manager = ValidatorManager(validator_config)
    potential_path = tmp_path / "potential.yace"
    potential_path.touch()

    with patch("pyacemaker.validator.manager.ReportGenerator") as MockReport:
        mock_instance = MockReport.return_value

        with (
            patch("pyacemaker.validator.manager.check_phonons") as mock_p,
            patch("pyacemaker.validator.manager.check_eos") as mock_e,
            patch("pyacemaker.validator.manager.check_elastic") as mock_el,
        ):
            mock_p.return_value = True
            mock_e.return_value = (100.0, "eos.png")
            mock_el.return_value = (True, {})

            result = manager.validate(potential_path, mock_atoms, output_dir=tmp_path)

            mock_instance.generate.assert_called_once()
            # Verify generated path
            args = mock_instance.generate.call_args
            assert args[0][1] == tmp_path / "validation_report.html"


def test_scenario_04_eon_execution(mock_atoms, eon_config, tmp_path):
    """Scenario 04: Verify EON execution."""
    wrapper = EONWrapper(eon_config)
    potential_path = Path("potential.yace")

    with patch("pyacemaker.dynamics.kmc.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0

        wrapper.run_search(mock_atoms, potential_path, work_dir=tmp_path)

        mock_run.assert_called_once()
        assert (tmp_path / "config.ini").exists()

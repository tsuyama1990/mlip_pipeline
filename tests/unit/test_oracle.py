from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.calculators.calculator import CalculatorError
from ase.io import write

from mlip_autopipec.config.config_model import OracleConfig
from mlip_autopipec.domain_models import Dataset

# This import will fail until implemented
from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle


@pytest.fixture
def mock_oracle_config(tmp_path: Path) -> OracleConfig:
    return OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=tmp_path / "pseudos",
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.04,
        scf_params={"mixing_beta": 0.7},
    )


@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_initialization(
    mock_espresso_cls: MagicMock, mock_oracle_config: OracleConfig
) -> None:
    """Test that EspressoOracle initializes correctly."""
    oracle = EspressoOracle(mock_oracle_config)
    assert oracle.config == mock_oracle_config


def test_espresso_oracle_initialization_bad_command(mock_oracle_config: OracleConfig) -> None:
    """Test security validation."""
    mock_oracle_config.command = "pw.x; rm -rf /"
    with pytest.raises(ValueError, match="Command contains invalid characters"):
        EspressoOracle(mock_oracle_config)


@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_label_success(
    mock_espresso_cls: MagicMock,
    mock_oracle_config: OracleConfig,
    tmp_path: Path,
) -> None:
    """Test successful labeling."""
    # Setup mocks
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc

    # Mock atoms and dataset
    dataset_path = tmp_path / "candidates.xyz"
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5])
    write(dataset_path, atoms)

    dataset = Dataset(file_path=dataset_path)

    # Run Oracle
    oracle = EspressoOracle(mock_oracle_config)
    labeled_dataset = oracle.label(dataset)

    # Verify calls
    assert mock_espresso_cls.called
    call_kwargs = mock_espresso_cls.call_args[1]
    assert call_kwargs["command"] == "pw.x"
    assert call_kwargs["pseudopotentials"] == {"Si": "Si.upf"}
    assert call_kwargs["kspacing"] == 0.04
    assert call_kwargs["mixing_beta"] == 0.7
    assert call_kwargs["tprnfor"] is True
    assert call_kwargs["tstress"] is True

    # Verify output
    assert labeled_dataset.file_path.exists()
    assert labeled_dataset.file_path != dataset_path


@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_recovery(
    mock_espresso_cls: MagicMock,
    mock_oracle_config: OracleConfig,
    tmp_path: Path,
) -> None:
    """Test self-healing recovery."""
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc

    # First call fails, second succeeds
    mock_calc.get_potential_energy.side_effect = [CalculatorError("Convergence failure"), 0.0]
    # We need to reset side_effect or handle subsequent calls for forces
    mock_calc.get_forces.return_value = [[0, 0, 0], [0, 0, 0]]

    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5])
    dataset_path = tmp_path / "candidates_recovery.xyz"
    write(dataset_path, atoms)

    dataset = Dataset(file_path=dataset_path)

    oracle = EspressoOracle(mock_oracle_config)
    labeled_dataset = oracle.label(dataset)

    assert labeled_dataset.file_path.exists()

    # Verify recovery was attempted
    assert mock_espresso_cls.call_count >= 2

    # Check parameters of second call
    calls = mock_espresso_cls.call_args_list
    first_call_params = calls[0][1]
    second_call_params = calls[1][1]

    assert first_call_params["mixing_beta"] == 0.7
    # Assuming recovery lowers it
    assert second_call_params["mixing_beta"] < 0.7


@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_all_retries_fail(
    mock_espresso_cls: MagicMock,
    mock_oracle_config: OracleConfig,
    tmp_path: Path,
) -> None:
    """Test behavior when all recovery attempts fail."""
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc

    # All calls fail
    mock_calc.get_potential_energy.side_effect = CalculatorError("Convergence failure")

    atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]], cell=[5, 5, 5])
    dataset_path = tmp_path / "candidates_fail.xyz"
    write(dataset_path, atoms)

    dataset = Dataset(file_path=dataset_path)

    oracle = EspressoOracle(mock_oracle_config)
    labeled_dataset = oracle.label(dataset)

    # Output file should exist but contain no atoms (since we append on success)
    assert labeled_dataset.file_path.exists()

    # Check that no atoms were written using file size
    assert labeled_dataset.file_path.stat().st_size == 0

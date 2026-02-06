from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from mlip_autopipec.config import OracleConfig
from mlip_autopipec.domain_models import Dataset
from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle


@pytest.fixture
def oracle_config(tmp_path: Path) -> OracleConfig:
    return OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=tmp_path / "pseudos",
        pseudopotentials={"Si": "Si.UPF"},
        kspacing=0.04,
    )

def test_espresso_oracle_init(oracle_config: OracleConfig, tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    oracle = EspressoOracle(oracle_config, work_dir)
    assert oracle.config == oracle_config
    assert oracle.work_dir == work_dir

@patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
@patch("mlip_autopipec.infrastructure.espresso.adapter.write")
@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_label_streaming(
    mock_espresso_cls: MagicMock,
    mock_write: MagicMock,
    mock_iread: MagicMock,
    oracle_config: OracleConfig,
    tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    oracle = EspressoOracle(oracle_config, work_dir)

    # Mock dataset
    dataset_path = tmp_path / "candidates.xyz"
    dataset_path.touch()
    dataset = Dataset(file_path=dataset_path)

    # Mock iread to return generator of atoms
    atoms1 = Atoms("Si", positions=[[0, 0, 0]])
    atoms2 = Atoms("Si", positions=[[1, 1, 1]])
    mock_iread.return_value = [atoms1, atoms2]

    # Mock calculator instance
    mock_calc = MagicMock()
    mock_calc.get_potential_energy.return_value = -10.0
    mock_calc.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_calc.get_stress.return_value = np.zeros(6)
    mock_espresso_cls.return_value = mock_calc

    # Run label
    oracle.label(dataset)

    # Verify iread called
    mock_iread.assert_called_with(dataset_path)

    # Verify calculator attached and calculated
    # Should be called for each atom
    assert mock_espresso_cls.call_count >= 2

    # Verify write called with append=True
    assert mock_write.call_count >= 2
    mock_write.assert_called_with(ANY, atoms2, format="extxyz", append=True)

@patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
@patch("mlip_autopipec.infrastructure.espresso.adapter.write") # Mock write to avoid FS
@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_label_recovery(
    mock_espresso_cls: MagicMock,
    mock_write: MagicMock,
    mock_iread: MagicMock,
    oracle_config: OracleConfig,
    tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    oracle = EspressoOracle(oracle_config, work_dir)

    atoms = Atoms("Si")
    mock_iread.return_value = [atoms]

    mock_calc = MagicMock()
    # First call to get_potential_energy fails, next succeeds (simulating retry with new params)
    mock_calc.get_potential_energy.side_effect = [Exception("Convergence failed"), 0.0]
    mock_calc.get_forces.return_value = np.array([[0.0, 0.0, 0.0]])
    mock_calc.get_stress.return_value = np.zeros(6)
    mock_espresso_cls.return_value = mock_calc

    input_file = tmp_path / "input.xyz"
    input_file.touch()
    oracle.label(Dataset(file_path=input_file))

    # Should have retried. get_potential_energy called twice.
    # Actually, if we re-instantiate, the new instance is called.
    # Since we return same mock_calc, call count accumulates on it.
    assert mock_calc.get_potential_energy.call_count == 2

@patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
@patch("mlip_autopipec.infrastructure.espresso.adapter.write")
@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_label_recovery_fails_all_attempts(
    mock_espresso_cls: MagicMock,
    mock_write: MagicMock,
    mock_iread: MagicMock,
    oracle_config: OracleConfig,
    tmp_path: Path
) -> None:
    work_dir = tmp_path / "work"
    oracle = EspressoOracle(oracle_config, work_dir)

    atoms = Atoms("Si")
    mock_iread.return_value = [atoms]

    mock_calc = MagicMock()
    # Always fails
    mock_calc.get_potential_energy.side_effect = Exception("Always fails")
    mock_espresso_cls.return_value = mock_calc

    input_file = tmp_path / "input.xyz"
    input_file.touch()

    # It logs exception but continues loop (caught in label loop)
    oracle.label(Dataset(file_path=input_file))

    # Verify write not called (since it failed)
    mock_write.assert_not_called()

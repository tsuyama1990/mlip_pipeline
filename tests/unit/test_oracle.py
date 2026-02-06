from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import CalculatorError
from ase.io import read, write

from mlip_autopipec.config import OracleConfig
from mlip_autopipec.domain_models import Dataset

# These imports are expected to exist in Phase 3
from mlip_autopipec.infrastructure.espresso.adapter import EspressoOracle


@pytest.fixture
def mock_oracle_config(tmp_path: Path) -> OracleConfig:
    # Create dummy pseudo dir
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    return OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.04,
    )


def test_espresso_oracle_initialization(mock_oracle_config: OracleConfig, tmp_path: Path) -> None:
    oracle = EspressoOracle(mock_oracle_config, tmp_path)
    assert oracle.config == mock_oracle_config
    assert oracle.work_dir == tmp_path


@patch("mlip_autopipec.infrastructure.espresso.adapter.logger")
@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_label_success(
    mock_espresso_cls: MagicMock,
    mock_logger: MagicMock,
    mock_oracle_config: OracleConfig,
    tmp_path: Path,
) -> None:
    """
    Test happy path for labeling.
    """
    oracle = EspressoOracle(mock_oracle_config, tmp_path)

    # Create input dataset
    input_file = tmp_path / "input.xyz"
    atoms = Atoms("Si2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[5, 5, 5], pbc=True)
    write(input_file, atoms)
    dataset = Dataset(file_path=input_file)

    # Mock calculator
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc
    mock_calc.results = {}

    def get_potential_energy_side_effect(*args: Any, **kwargs: Any) -> float:
        mock_calc.results["energy"] = -200.0
        return -200.0

    mock_calc.get_potential_energy.side_effect = get_potential_energy_side_effect
    mock_calc.get_forces.return_value = np.zeros((2, 3))
    mock_calc.get_stress.return_value = np.zeros(6)

    # Run label
    result_dataset = oracle.label(dataset)

    # Verify result
    assert result_dataset.file_path.exists()
    # Check that output is not the same as input
    assert result_dataset.file_path != input_file

    # Verify results content
    labeled_atoms = read(result_dataset.file_path)
    # read() returns Atoms or list[Atoms]. Since we wrote one, it might be Atoms.
    # But label() writes extxyz which read() might return as list if index=':'.
    # Let's handle both or check generic read.
    if isinstance(labeled_atoms, list):
        labeled_atoms = labeled_atoms[0]

    # Check energy using get_potential_energy() which covers both info dict and attached calculator
    # read() from extxyz might put energy in info OR attach SinglePointCalculator
    energy = labeled_atoms.info.get("energy")
    if energy is None and labeled_atoms.calc:
        energy = labeled_atoms.get_potential_energy()  # type: ignore[no-untyped-call]

    assert energy == -200.0


@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_recovery(
    mock_espresso_cls: MagicMock, mock_oracle_config: OracleConfig, tmp_path: Path
) -> None:
    """
    Test that recovery strategy is triggered on failure.
    """
    oracle = EspressoOracle(mock_oracle_config, tmp_path)

    # Create input
    input_file = tmp_path / "input_recovery.xyz"
    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    write(input_file, atoms)
    dataset = Dataset(file_path=input_file)

    # Mock calculator to fail first, succeed second
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc

    # Side effect: first call raises, second returns value
    mock_calc.get_potential_energy.side_effect = [
        CalculatorError("Convergence failed"),
        -100.0,
    ]
    mock_calc.get_forces.return_value = np.zeros((1, 3))
    mock_calc.get_stress.return_value = np.zeros(6)

    # Run label
    result_dataset = oracle.label(dataset)

    # Verify result exists
    assert result_dataset.file_path.exists()

    # Verify that get_potential_energy was called twice
    assert mock_calc.get_potential_energy.call_count == 2

    # Verify that parameters were changed (recovery logic)
    # First call: default params
    # Second call: modified params (e.g. mixing_beta changed)
    # We can inspect the calculator instance attributes if the adapter updates them


def test_espresso_oracle_dangerous_command(tmp_path: Path) -> None:
    """
    Test that dangerous commands are rejected.
    """
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    config = OracleConfig(
        type="espresso",
        command="pw.x; rm -rf /",
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.04,
    )

    with pytest.raises(ValueError, match="Security check failed"):
        EspressoOracle(config, tmp_path)

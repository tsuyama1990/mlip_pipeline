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
from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy


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
    # Customize recipes for this test to verify they are used
    mock_oracle_config.recovery_recipes = [{"mixing_beta": 0.5}]

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

    # Verify that the second call used the custom recipe
    # Espresso init arguments on second call should contain mixing_beta=0.5
    # The first call uses defaults. The second uses mixing_beta=0.5
    assert mock_espresso_cls.call_count == 2

    # Check call args of second call
    _, kwargs = mock_espresso_cls.call_args_list[1]
    input_data = kwargs.get("input_data", {})
    assert input_data.get("mixing_beta") == 0.5


def test_espresso_oracle_dangerous_command_whitelist(tmp_path: Path) -> None:
    """
    Test that dangerous commands or unwhitelisted executables are rejected.
    """
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()

    # 1. Allowed executable
    config = OracleConfig(
        type="espresso",
        command="pw.x",
        pseudo_dir=pseudo_dir,
        pseudopotentials={"Si": "Si.upf"},
        kspacing=0.04,
    )
    EspressoOracle(config, tmp_path) # Should pass

    # 2. Allowed executable with args
    config.command = "mpirun -np 4 pw.x"
    EspressoOracle(config, tmp_path) # Should pass

    # 3. Disallowed executable (e.g. rm)
    config.command = "rm -rf /"
    with pytest.raises(ValueError, match="Security check failed"):
        EspressoOracle(config, tmp_path)

    # 4. Dangerous chars (;)
    config.command = "pw.x; rm -rf /"
    with pytest.raises(ValueError, match="Security check failed"):
        EspressoOracle(config, tmp_path)


@patch("mlip_autopipec.infrastructure.espresso.adapter.write")
@patch("mlip_autopipec.infrastructure.espresso.adapter.iread")
@patch("mlip_autopipec.infrastructure.espresso.adapter.Espresso")
def test_espresso_oracle_batching(
    mock_espresso_cls: MagicMock,
    mock_iread: MagicMock,
    mock_write: MagicMock,
    mock_oracle_config: OracleConfig,
    tmp_path: Path
) -> None:
    """
    Test that the oracle processes and writes in batches (checks that write is called for lists).
    Actually, we want to check that it writes occasionally, not necessarily lists if using append.
    But implementing batch writes means passing a list to write().
    """
    oracle = EspressoOracle(mock_oracle_config, tmp_path)
    dataset = Dataset(file_path=tmp_path / "dummy.xyz")
    (tmp_path / "dummy.xyz").touch()

    # Mock input: 5 atoms
    atoms = Atoms("H")
    mock_iread.return_value = [atoms] * 5

    # Mock calculator success
    mock_calc = MagicMock()
    mock_espresso_cls.return_value = mock_calc
    mock_calc.get_potential_energy.return_value = -1.0

    # If we implement batch size of e.g. 2, we expect write to be called 3 times (2, 2, 1) or similar.
    # For now, let's just assume we want to verify it finishes.
    # But to test batching specifically, we need to inspect implementation or spy on write.
    # Let's assume implementation uses a batch size. We can verify write calls receive lists.

    oracle.label(dataset)

    # Verify write was called with list of atoms (if batched) or individual (if not)
    # The requirement is to batch writes.
    # So write(file, list_of_atoms, append=True)
    assert mock_write.called
    args, _ = mock_write.call_args
    # Check if second arg is list
    if len(args) > 1:
        assert isinstance(args[1], list)


def test_recovery_strategy_config_injection() -> None:
    recipes = [{"beta": 0.1}]
    strategy = RecoveryStrategy(recipes)
    assert strategy.get_recipe(0) == {"beta": 0.1}
    assert strategy.get_recipe(1) is None

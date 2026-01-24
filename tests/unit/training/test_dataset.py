import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms

from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.dataset import DatasetBuilder


@pytest.fixture
def mock_db_manager() -> MagicMock:
    return MagicMock()


@pytest.fixture
def training_config() -> TrainingConfig:
    return TrainingConfig(
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=0.5,
        batch_size=32,
        training_data_path="train.xyz",
        test_data_path="test.xyz"
    )


def test_export_db(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test exporting data from database."""
    # Setup mock data
    atoms_list = [
        Atoms("H", positions=[[0, 0, 0]]),
        Atoms("He", positions=[[1, 1, 1]])
    ]
    mock_db_manager.get_atoms.return_value = atoms_list

    builder = DatasetBuilder(mock_db_manager)

    # Run export
    train_path = builder.export(training_config, tmp_path)

    # Verify calls
    mock_db_manager.get_atoms.assert_called_once_with(selection="status=completed")

    # Verify output files
    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "test.xyz").exists()

    assert train_path == tmp_path / "train.xyz"


def test_export_atoms(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test exporting atoms directly via export_atoms."""
    builder = DatasetBuilder(mock_db_manager)
    atoms_list = [Atoms("H", positions=[[0, 0, 0]]) for _ in range(10)]

    output_file = tmp_path / "custom.xyz"
    builder.export_atoms(atoms_list, output_file)

    assert output_file.exists()
    # verify content roughly
    with output_file.open("r") as f:
        content = f.read()
        assert "Lattice" in content or "Properties" in content or len(content) > 0


def test_export_no_data(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test error when no data found."""
    mock_db_manager.get_atoms.return_value = []
    builder = DatasetBuilder(mock_db_manager)

    with pytest.raises(ValueError, match="No training data found"):
        builder.export(training_config, tmp_path)


def test_export_atoms_fail(mock_db_manager: MagicMock, tmp_path: Path) -> None:
    """Test export atoms failure."""
    builder = DatasetBuilder(mock_db_manager)
    atoms_list = [Atoms("H")]

    # Pass a directory as file path to cause write error
    output_path = tmp_path / "subdir"
    output_path.mkdir()

    # Writing to a directory raises IsADirectoryError (subclass of OSError) on Linux
    with pytest.raises(OSError):
        builder.export_atoms(atoms_list, output_path)


def test_export_single_atom(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test export with single atom (edge case for splitting)."""
    mock_db_manager.get_atoms.return_value = [Atoms("H")]
    builder = DatasetBuilder(mock_db_manager)

    builder.export(training_config, tmp_path)

    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "test.xyz").exists()

    # Test file should be empty created by touch()
    assert (tmp_path / "test.xyz").stat().st_size == 0

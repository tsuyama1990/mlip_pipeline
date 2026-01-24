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


def test_export_db_streaming(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test exporting data using streaming select."""
    # Setup mock data generator
    atoms_list = [
        Atoms("H", positions=[[0, 0, 0]]),
        Atoms("He", positions=[[1, 1, 1]])
    ]

    def mock_select(**kwargs):
        yield from atoms_list

    mock_db_manager.select.side_effect = mock_select

    builder = DatasetBuilder(mock_db_manager)

    # Run export
    train_path = builder.export(training_config, tmp_path)

    # Verify calls
    mock_db_manager.select.assert_called_once_with(selection="status=completed")

    # Verify output files exist
    assert (tmp_path / "train.xyz").exists()
    assert (tmp_path / "test.xyz").exists()

    # Verify at least one file has content (probabilistic, but with 2 atoms seed 42, likely split)
    s1 = (tmp_path / "train.xyz").stat().st_size
    s2 = (tmp_path / "test.xyz").stat().st_size
    assert s1 > 0 or s2 > 0


def test_export_no_data(mock_db_manager: MagicMock, training_config: TrainingConfig, tmp_path: Path) -> None:
    """Test error when no data found."""
    def mock_select(**kwargs):
        return
        yield

    mock_db_manager.select.side_effect = mock_select
    builder = DatasetBuilder(mock_db_manager)

    with pytest.raises(ValueError, match="No training data found"):
        builder.export(training_config, tmp_path)


def test_export_atoms_fail(mock_db_manager: MagicMock, tmp_path: Path) -> None:
    """Test export atoms failure."""
    builder = DatasetBuilder(mock_db_manager)
    atoms_list = [Atoms("H")]

    output_path = tmp_path / "subdir"
    output_path.mkdir()

    with pytest.raises(OSError):
        builder.export_atoms(atoms_list, output_path)

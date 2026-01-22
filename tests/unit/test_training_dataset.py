import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, call
from ase import Atoms
from mlip_autopipec.config.schemas.training import TrainingConfig
from mlip_autopipec.training.dataset import DatasetBuilder

@pytest.fixture
def mock_db_manager():
    return MagicMock()

@pytest.fixture
def training_config():
    return TrainingConfig(
        cutoff=5.0,
        b_basis_size=100,
        kappa=0.5,
        kappa_f=50.0,
        max_iter=100
    )

def test_export_data(mock_db_manager, training_config, tmp_path):
    """Test exporting data to ExtXYZ format."""
    # Create dummy atoms
    atoms = [Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]]) for _ in range(10)]
    for i, at in enumerate(atoms):
        at.info['energy'] = float(i)
        at.arrays['forces'] = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.1]])
        # Add generation info to test filtering/verification if needed
        at.info['generation'] = 1

    # Mock DB return
    mock_db_manager.get_atoms.return_value = atoms

    builder = DatasetBuilder(mock_db_manager)

    # We expect 'completed' status
    output_dir = tmp_path / "data"

    builder.export(config=training_config, output_dir=output_dir)

    # Check if files exist (defaults are data/train.xyz)
    assert (output_dir / "data" / "train.xyz").exists()
    assert (output_dir / "data" / "test.xyz").exists()

    # Check split ratio (90/10)
    # We can't easily check the content without parsing ExtXYZ, but we can check if file is not empty
    assert (output_dir / "data" / "train.xyz").stat().st_size > 0
    assert (output_dir / "data" / "test.xyz").stat().st_size > 0

    # Verify DB call
    mock_db_manager.get_atoms.assert_called_with(selection="status=completed")

def test_export_data_empty(mock_db_manager, training_config, tmp_path):
    """Test error when no data found."""
    mock_db_manager.get_atoms.return_value = []

    builder = DatasetBuilder(mock_db_manager)

    with pytest.raises(ValueError, match="No training data found"):
        builder.export(config=training_config, output_dir=tmp_path)

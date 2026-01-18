import gzip
import pickle
from pathlib import Path
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from mlip_autopipec.config.schemas.training import TrainConfig
from mlip_autopipec.training.dataset import DatasetBuilder


@pytest.fixture
def mock_db():
    db = Mock()
    # Mock get_atoms to return a list of atoms

    atoms_list = []
    for i in range(10):
        at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.0 + i * 0.1]])
        at.calc = SinglePointCalculator(at, energy=-10.0, forces=np.zeros((2, 3)))
        at.info = {}  # Ensure info dict exists
        atoms_list.append(at)

    db.get_atoms.return_value = atoms_list
    return db


def test_dataset_fetch_valid(mock_db):
    """Test fetching valid data."""
    builder = DatasetBuilder(mock_db)
    data = builder.fetch_data()
    assert len(data) == 10
    assert isinstance(data[0], Atoms)


def test_dataset_fetch_invalid_type(mock_db):
    """Test validation when DB returns non-Atoms objects."""
    # This should fail pydantic validation in TrainingBatch
    mock_db.get_atoms.return_value = ["not_an_atom_object"]
    builder = DatasetBuilder(mock_db)

    # We expect pydantic TypeError or Validation Error wrapped
    with pytest.raises((TypeError, ValueError)):
        builder.fetch_data()


def test_dataset_export_delta_learning(mock_db, tmp_path):
    builder = DatasetBuilder(mock_db)
    config = TrainConfig(enable_delta_learning=True, test_fraction=0.2)

    output_path = builder.export(config, tmp_path)

    assert output_path.exists()
    assert output_path.name == "train.pckl.gzip"

    # Check content
    with gzip.open(output_path, "rb") as f:
        atoms_train = pickle.load(f)

    assert len(atoms_train) == 8  # 10 * (1 - 0.2)

    # Check if delta learning was applied
    # Original E = -10.0. ZBL E > 0.
    # New E should be -10.0 - ZBL. So it should be < -10.0.
    assert atoms_train[0].info["energy"] < -10.0
    assert "zbl_energy" in atoms_train[0].info


def test_dataset_export_no_delta(mock_db, tmp_path):
    builder = DatasetBuilder(mock_db)
    config = TrainConfig(enable_delta_learning=False, test_fraction=0.0)

    output_path = builder.export(config, tmp_path)

    with gzip.open(output_path, "rb") as f:
        atoms_train = pickle.load(f)

    assert len(atoms_train) == 10
    assert atoms_train[0].info["energy"] == -10.0
    assert "zbl_energy" not in atoms_train[0].info


def test_force_masking(mock_db, tmp_path):
    # Modify one atom to have force_mask
    atoms_list = mock_db.get_atoms.return_value
    at_masked = atoms_list[0]
    at_masked.set_array("force_mask", np.array([1, 0]))

    builder = DatasetBuilder(mock_db)
    config = TrainConfig(enable_delta_learning=False)

    output_path = builder.export(config, tmp_path)

    with gzip.open(output_path, "rb") as f:
        atoms_train = pickle.load(f)

    # Find the masked one
    found = False
    for at in atoms_train:
        if "weights" in at.arrays:
            assert np.array_equal(at.arrays["weights"], np.array([1.0, 0.0]))
            found = True

    assert found

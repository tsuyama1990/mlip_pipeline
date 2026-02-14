"""Tests for Dataset Manager."""

import gzip
import pickle
from pathlib import Path

import pytest
from ase import Atoms

from pyacemaker.oracle.dataset import DatasetManager


def test_save_load_dataset(tmp_path: Path) -> None:
    """Test saving and loading a dataset."""
    dataset_path = tmp_path / "dataset.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    data = [atoms1, atoms2]

    manager.save(data, dataset_path)
    assert dataset_path.exists()

    loaded_data = manager.load(dataset_path)
    assert len(loaded_data) == 2
    # ase.Atoms.get_chemical_formula is untyped
    assert loaded_data[0].get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]
    assert loaded_data[1].get_chemical_formula() == "O2"  # type: ignore[no-untyped-call]


def test_load_iter(tmp_path: Path) -> None:
    """Test loading a dataset using iterator."""
    dataset_path = tmp_path / "iter_dataset.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    data = [atoms1, atoms2]

    manager.save(data, dataset_path)
    assert dataset_path.exists()

    # Load as iterator
    iterator = manager.load_iter(dataset_path)
    loaded_data = list(iterator)
    assert len(loaded_data) == 2
    assert loaded_data[0].get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]
    assert loaded_data[1].get_chemical_formula() == "O2"  # type: ignore[no-untyped-call]


def test_load_iter_sequential(tmp_path: Path) -> None:
    """Test loading a dataset with sequentially dumped objects."""
    dataset_path = tmp_path / "seq_dataset.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2")
    atoms2 = Atoms("O2")

    # Manually dump sequentially
    with gzip.open(dataset_path, "wb") as f:
        pickle.dump(atoms1, f)
        pickle.dump(atoms2, f)

    iterator = manager.load_iter(dataset_path)
    loaded_data = list(iterator)
    assert len(loaded_data) == 2
    assert loaded_data[0].get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]
    assert loaded_data[1].get_chemical_formula() == "O2"  # type: ignore[no-untyped-call]


def test_load_non_existent_file() -> None:
    """Test loading a non-existent file."""
    manager = DatasetManager()
    with pytest.raises(FileNotFoundError):
        manager.load(Path("non_existent.pckl.gzip"))
    with pytest.raises(FileNotFoundError):
        next(manager.load_iter(Path("non_existent.pckl.gzip")))


def test_load_invalid_type(tmp_path: Path) -> None:
    """Test loading a file with invalid data type."""
    path = tmp_path / "invalid.pckl.gzip"
    with gzip.open(path, "wb") as f:
        pickle.dump({"not": "a list"}, f)

    manager = DatasetManager()
    with pytest.raises(TypeError, match="must contain a list"):
        manager.load(path)

"""Tests for Dataset Manager."""

import gzip
import pickle
import struct
from pathlib import Path

import pytest
from ase import Atoms

from pyacemaker.oracle.dataset import DatasetManager


def test_save_load_iter(tmp_path: Path) -> None:
    """Test saving and loading a dataset using stream-friendly methods."""
    dataset_path = tmp_path / "stream_dataset.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    data = [atoms1, atoms2]

    # Use save_iter (recommended)
    manager.save_iter(data, dataset_path)
    assert dataset_path.exists()

    # Load as iterator
    iterator = manager.load_iter(dataset_path, verify=False)
    loaded_data = list(iterator)
    assert len(loaded_data) == 2
    assert loaded_data[0].get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]
    assert loaded_data[1].get_chemical_formula() == "O2"  # type: ignore[no-untyped-call]


def test_load_iter_legacy_list_rejection(tmp_path: Path) -> None:
    """Test that loading a dataset saved as a single list raises ValueError (invalid header)."""
    dataset_path = tmp_path / "legacy_dataset.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2")
    data = [atoms1]

    # Save as single list pickle
    with gzip.open(dataset_path, "wb") as f:
        pickle.dump(data, f)

    # load_iter should reject it due to header validation failure (or size)
    # The header validation is stricter now, so we expect ValueError (corrupted or too large)
    iterator = manager.load_iter(dataset_path, verify=False)
    with pytest.raises(ValueError, match="Object size .* exceeds limit"):
        next(iterator)


def test_load_non_existent_file() -> None:
    """Test loading a non-existent file."""
    manager = DatasetManager()
    with pytest.raises(FileNotFoundError):
        next(manager.load_iter(Path("non_existent.pckl.gzip")))


def test_load_iter_corrupted(tmp_path: Path) -> None:
    """Test loading a corrupted file stops gracefully."""
    dataset_path = tmp_path / "corrupted.pckl.gzip"
    manager = DatasetManager()

    atoms1 = Atoms("H2")

    # Manually write a valid Framed Pickle object
    obj_bytes = pickle.dumps(atoms1)
    size = len(obj_bytes)

    with gzip.open(dataset_path, "wb") as f:
        # Valid header and payload
        f.write(struct.pack(">Q", size))
        f.write(obj_bytes)

        # Write garbage (short, to trigger partial read handling)
        f.write(b"short")

    iterator = manager.load_iter(dataset_path, verify=False)
    # Should read the first atom
    first = next(iterator)
    assert first.get_chemical_formula() == "H2"  # type: ignore[no-untyped-call]

    # Should stop iteration (StopIteration implicitly)
    with pytest.raises(StopIteration):
        next(iterator)


def test_load_iter_empty(tmp_path: Path) -> None:
    """Test loading an empty file."""
    dataset_path = tmp_path / "empty.pckl.gzip"
    # Create empty gzip file
    with gzip.open(dataset_path, "wb"):
        pass

    manager = DatasetManager()
    iterator = manager.load_iter(dataset_path, verify=False)
    assert list(iterator) == []

from pathlib import Path

import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def test_dataset_init_confinement(tmp_path: Path) -> None:
    """Test path confinement checks."""
    root = tmp_path / "data"
    root.mkdir()

    # Valid
    d = Dataset(root / "dataset.jsonl", root_dir=root)
    assert d.path == root / "dataset.jsonl"

    # Invalid (outside root)
    outside = tmp_path / "outside.jsonl"
    with pytest.raises(ValueError, match="outside the allowed root directory"):
        Dataset(outside, root_dir=root)


def test_dataset_append_and_read(tmp_path: Path) -> None:
    """Test appending and reading structures."""
    root = tmp_path
    d = Dataset(root / "test.jsonl", root_dir=root)

    # Create dummy structure
    import numpy as np
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=-1.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

    # Append
    d.append([s])
    assert len(d) == 1

    # Read
    read_s = next(iter(d))
    assert read_s.energy == -1.0


def test_dataset_invalid_path(tmp_path: Path) -> None:
    """Test invalid path characters."""
    # Note: On some systems/python versions, pathlib.resolve raises ValueError for null bytes
    # with 'embedded null character', on others our check catches it.
    with pytest.raises(ValueError, match="null byte|null character"):
        Dataset(Path("test\0.jsonl"), root_dir=tmp_path)

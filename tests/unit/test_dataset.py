from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models import Structure


def create_dummy_structure(idx: int) -> Structure:
    return Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        energy=-1.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3)),
        properties={"idx": idx},
    )


def test_dataset_init(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    dataset = Dataset(path)
    assert len(dataset) == 0
    assert dataset.path == path.resolve()
    assert dataset.meta_path.exists()


def test_dataset_append_streaming(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    dataset = Dataset(path)

    structures = [create_dummy_structure(i) for i in range(10)]
    dataset.append(structures, batch_size=2)

    assert len(dataset) == 10

    # Verify content
    loaded = list(dataset)
    assert len(loaded) == 10
    assert loaded[0].properties is not None
    assert loaded[0].properties["idx"] == 0
    assert loaded[9].properties is not None
    assert loaded[9].properties["idx"] == 9


def test_dataset_security_check(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()

    # Path inside root
    safe_path = root / "data.jsonl"
    Dataset(safe_path, root_dir=root)

    # Path outside root
    unsafe_path = tmp_path / "outside.jsonl"
    with pytest.raises(ValueError, match="outside root directory"):
        Dataset(unsafe_path, root_dir=root)


def test_dataset_corrupt_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "corrupt.jsonl"

    s = create_dummy_structure(0)
    valid_json = s.model_dump_json()

    path.write_text(f"{valid_json}\nBROKEN_JSON\n{valid_json}")

    dataset = Dataset(path)

    items = list(dataset)
    assert len(items) == 2
    assert "Corrupt dataset line" in caplog.text


def test_dataset_memory_usage(tmp_path: Path) -> None:
    """
    Verify that iterating over a large dataset does not consume excessive memory.
    This test creates a file with 1000 items and iterates.
    Ideally we would measure memory, but here we just ensure it runs and we verify logic (by code inspection mainly).
    We can check that __iter__ returns a generator.
    """
    path = tmp_path / "large.jsonl"
    dataset = Dataset(path)

    # Write directly to file to simulate large dataset without creating objects in memory first
    s = create_dummy_structure(0)
    line = s.model_dump_json() + "\n"

    with path.open("w") as f:
        for _ in range(1000):
            f.write(line)

    iterator = iter(dataset)
    # Check it's a generator/iterator
    assert hasattr(iterator, "__next__")

    # Consume
    count = 0
    for _ in iterator:
        count += 1
    assert count == 1000

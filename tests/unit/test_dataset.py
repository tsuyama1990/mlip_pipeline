from pathlib import Path

import numpy as np
import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def test_dataset_initialization(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path)

    assert dataset_path.exists()
    assert (tmp_path / "dataset.meta.json").exists()
    assert len(dataset) == 0


def test_dataset_append_and_read(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path)

    structures = []
    for _ in range(3):
        structures.append(
            Structure(
                positions=np.zeros((1, 3)),
                atomic_numbers=np.array([1]),
                cell=np.eye(3),
                pbc=np.array([True, True, True]),
            )
        )

    dataset.append(structures)
    assert len(dataset) == 3

    loaded_structures = list(dataset)
    assert len(loaded_structures) == 3
    for s in loaded_structures:
        assert isinstance(s, Structure)


def test_dataset_persistence(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset1 = Dataset(dataset_path)

    structures = [
        Structure(
            positions=np.zeros((1, 3)),
            atomic_numbers=np.array([1]),
            cell=np.eye(3),
            pbc=np.array([True, True, True]),
        )
    ]
    dataset1.append(structures)

    dataset2 = Dataset(dataset_path)
    assert len(dataset2) == 1


def test_dataset_streaming_append(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path)

    structures = []
    for _ in range(15):
        structures.append(
            Structure(
                positions=np.zeros((1, 3)),
                atomic_numbers=np.array([1]),
                cell=np.eye(3),
                pbc=np.array([True, True, True]),
            )
        )

    # Should write all 15 structures
    dataset.append(structures)
    assert len(dataset) == 15
    assert len(list(dataset)) == 15


def test_dataset_malformed_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path)

    # Write some good data
    good_structure = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
    )
    dataset.append([good_structure])

    # Corrupt the file
    with dataset_path.open("a") as f:
        f.write("{invalid_json}\n")
        f.write(good_structure.model_dump_json() + "\n")

    loaded = list(dataset)
    # Should skip the bad line and load 2 good structures
    assert len(loaded) == 2
    assert "Skipping malformed line" in caplog.text


def test_dataset_invalid_meta(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset = Dataset(dataset_path)
    meta_path = tmp_path / "dataset.meta.json"

    # Write invalid JSON
    with meta_path.open("w") as f:
        f.write("[]")  # List instead of dict

    with pytest.raises(TypeError, match="Metadata file must contain a JSON object"):
        len(dataset)

    with meta_path.open("w") as f:
        f.write('{"count": "invalid"}')  # Count not int

    with pytest.raises(TypeError, match="Metadata 'count' must be an integer"):
        len(dataset)

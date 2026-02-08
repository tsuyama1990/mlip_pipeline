from pathlib import Path

import numpy as np

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

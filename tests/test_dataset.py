from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models import Structure


def test_dataset_initialization(tmp_path: Path) -> None:
    ds_path = tmp_path / "data.jsonl"
    ds = Dataset(ds_path)
    assert ds_path.exists()
    assert (tmp_path / "data.meta.json").exists()
    assert ds.count() == 0

def test_dataset_append_one(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "data.jsonl")
    s = Structure(
        atomic_numbers=[1],
        positions=[[0,0,0]],
        cell=[[1,0,0],[0,1,0],[0,0,1]],
        pbc=[True, True, True],
        energy=0.0, forces=[[0,0,0]]
    )
    ds.append([s])
    assert ds.count() == 1

    # Check file content
    with ds.filepath.open() as f:
        lines = f.readlines()
        assert len(lines) == 1
        assert "atomic_numbers" in lines[0]

def test_dataset_streaming_iteration(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "stream.jsonl")
    # Add 10 structures
    structures = [
        Structure(
            atomic_numbers=[i],
            positions=[[0,0,0]],
            cell=[[1,0,0],[0,1,0],[0,0,1]],
            pbc=[True, True, True],
            energy=float(i), forces=[[0,0,0]]
        ) for i in range(10)
    ]
    ds.append(structures)

    # Iterate and verify we get them back in order
    retrieved = []
    for s in ds:
        retrieved.append(s)

    assert len(retrieved) == 10
    assert retrieved[0].energy == 0.0
    assert retrieved[9].energy == 9.0

def test_dataset_validation_error(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "invalid.jsonl")
    s = Structure(
        atomic_numbers=[1],
        positions=[[0,0,0]],
        cell=[[1,0,0],[0,1,0],[0,0,1]],
        pbc=[True, True, True]
        # Missing energy/forces
    )
    with pytest.raises(ValueError, match="Structure must have energy"):
        ds.append([s])

    # Should not have incremented count
    assert ds.count() == 0
    with ds.filepath.open() as f:
        assert len(f.readlines()) == 0

def test_dataset_io_error_handling(tmp_path: Path) -> None:
    ds = Dataset(tmp_path / "mock_io.jsonl")
    s = Structure(
        atomic_numbers=[1],
        positions=[[0,0,0]],
        cell=[[1,0,0],[0,1,0],[0,0,1]],
        pbc=[True, True, True],
        energy=0.0, forces=[[0,0,0]]
    )

    with patch.object(Path, 'open', side_effect=OSError("Disk full")), \
         pytest.raises(RuntimeError, match="Failed to append to dataset"):
         ds.append([s])

def test_dataset_streaming_append(tmp_path: Path) -> None:
    """Verify that append consumes iterator lazily."""
    ds = Dataset(tmp_path / "stream_append.jsonl")

    def generator() -> Iterator[Structure]:
        for i in range(5):
            yield Structure(
                atomic_numbers=[1],
                positions=[[0,0,0]],
                cell=[[1,0,0],[0,1,0],[0,0,1]],
                pbc=[True, True, True],
                energy=float(i), forces=[[0,0,0]]
            )

    # Should consume the generator
    ds.append(generator())
    assert ds.count() == 5

from pathlib import Path

from mlip_autopipec.core.dataset import Dataset
from mlip_autopipec.domain_models.structure import Structure


def test_dataset_streaming_append(tmp_path: Path) -> None:
    """Verify that dataset append works for iterators."""
    root = tmp_path
    d = Dataset(root / "stream.jsonl", root_dir=root)

    def structure_generator():
        import numpy as np
        for i in range(5):
            yield Structure(
                positions=np.zeros((1, 3)),
                atomic_numbers=np.array([1]),
                cell=np.eye(3),
                pbc=np.array([True, True, True]),
                energy=float(i),
                forces=np.zeros((1, 3)),
                stress=np.zeros((3, 3))
            )

    # Append generator
    d.append(structure_generator())
    assert len(d) == 5

    # Read back
    energies = [s.energy for s in d]
    assert energies == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_dataset_iter_batches(tmp_path: Path) -> None:
    """Test batch iteration."""
    root = tmp_path
    d = Dataset(root / "batch.jsonl", root_dir=root)

    import numpy as np
    s = Structure(
        positions=np.zeros((1, 3)),
        atomic_numbers=np.array([1]),
        cell=np.eye(3),
        pbc=np.array([True, True, True]),
        energy=0.0,
        forces=np.zeros((1, 3)),
        stress=np.zeros((3, 3))
    )

    # Append 25 items
    d.append([s] * 25)

    batches = list(d.iter_batches(batch_size=10))
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5

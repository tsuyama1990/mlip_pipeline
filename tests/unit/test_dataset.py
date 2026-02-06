from ase import Atoms

from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata


def test_dataset_add_batch() -> None:
    dataset = Dataset()
    atoms = Atoms("H")
    meta1 = StructureMetadata(structure=atoms, source="initial", generation_method="random")
    meta2 = StructureMetadata(structure=atoms, source="initial", generation_method="random")

    dataset.add_batch([meta1, meta2])
    assert len(dataset) == 2


def test_dataset_stream() -> None:
    dataset = Dataset()
    atoms = Atoms("H")
    meta1 = StructureMetadata(structure=atoms, source="initial", generation_method="random")
    meta2 = StructureMetadata(structure=atoms, source="initial", generation_method="random")
    dataset.add_batch([meta1, meta2])

    streamed = list(dataset.stream())
    assert len(streamed) == 2
    assert streamed[0] == meta1
    assert streamed[1] == meta2


def test_dataset_encapsulation() -> None:
    dataset = Dataset()
    # Ensure no public list exposed
    assert not hasattr(dataset, "structures")

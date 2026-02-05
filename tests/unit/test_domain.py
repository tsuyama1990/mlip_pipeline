import pytest
from ase import Atoms

from mlip_autopipec.domain_models import Dataset, StructureMetadata


def test_valid_structure_metadata() -> None:
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 1, 0]])
    meta = StructureMetadata(structure=atoms, source="test", generation_method="random")
    assert meta.structure == atoms


def test_invalid_structure_type() -> None:
    # It raises TypeError because we raise TypeError in validator
    with pytest.raises(TypeError):
        StructureMetadata(
            structure="not an atoms object", source="test", generation_method="random"
        )


def test_dataset_initialization() -> None:
    atoms = Atoms("H2")
    meta = StructureMetadata(structure=atoms, source="src", generation_method="gen")
    dataset = Dataset(structures=[meta])
    assert len(dataset.structures) == 1
    assert dataset.structures[0] == meta

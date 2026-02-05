import pytest
from ase import Atoms
from pydantic import ValidationError

from domain_models import Dataset, StructureMetadata


def test_structure_metadata_valid():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(structure=atoms, source="test", generation_method="manual")
    assert meta.source == "test"
    assert isinstance(meta.structure, Atoms)


def test_structure_metadata_invalid_structure():
    with pytest.raises(ValidationError) as excinfo:
        StructureMetadata(structure="not atoms", source="test", generation_method="manual")
    # Verify the Pydantic validation error
    assert "Input should be an instance of Atoms" in str(excinfo.value)


def test_dataset_valid():
    atoms = Atoms("H2")
    meta = StructureMetadata(structure=atoms, source="test", generation_method="manual")
    dataset = Dataset(structures=[meta], description="Test dataset")
    assert len(dataset.structures) == 1
    assert dataset.structures[0] == meta

from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Dataset, StructureMetadata


def test_structure_metadata_valid() -> None:
    atoms = Atoms("H2O")
    meta = StructureMetadata(structure=atoms, energy=-10.5, iteration=1)
    assert meta.structure == atoms
    assert meta.energy == -10.5
    assert meta.iteration == 1
    assert meta.forces is None

def test_dataset_initialization() -> None:
    atoms1 = Atoms("H")
    atoms2 = Atoms("O")
    meta1 = StructureMetadata(structure=atoms1)
    meta2 = StructureMetadata(structure=atoms2)

    dataset = Dataset(structures=[meta1, meta2])
    assert len(dataset.structures) == 2
    assert dataset.structures[0].structure == atoms1

def test_dataset_mutually_exclusive() -> None:
    """Test that Dataset cannot have both file_path and structures."""
    atoms = Atoms("H")
    meta = StructureMetadata(structure=atoms)

    # Valid: only structures
    Dataset(structures=[meta])

    # Valid: only file_path
    Dataset(file_path=Path("dataset.xyz"))

    # Valid: file_path and empty structures (default)
    Dataset(file_path=Path("dataset.xyz"), structures=[])

    # Case: Invalid - both provided
    with pytest.raises(ValidationError) as excinfo:
        Dataset(file_path=Path("dataset.xyz"), structures=[meta])
    assert "Dataset cannot have both 'file_path' and non-empty 'structures'" in str(excinfo.value)

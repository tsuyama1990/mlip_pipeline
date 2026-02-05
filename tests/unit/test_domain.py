import pytest
from ase import Atoms
from pydantic import ValidationError

from src.domain_models.dataset import Dataset
from src.domain_models.structures import StructureMetadata


def test_valid_structure_metadata() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(structure=atoms, source="test", generation_method="manual")
    assert meta.structure == atoms


def test_invalid_structure_type() -> None:
    # Pydantic wraps the TypeError in ValidationError
    with pytest.raises(ValidationError):
        StructureMetadata(
            structure="not atoms",  # type: ignore[arg-type]
            source="test",
            generation_method="manual",
        )


def test_dataset() -> None:
    ds = Dataset()
    assert len(ds) == 0
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(structure=atoms, source="test", generation_method="manual")
    ds.add(meta)
    assert len(ds) == 1
    assert ds.structures[0] == meta

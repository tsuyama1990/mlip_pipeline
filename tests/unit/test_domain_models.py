import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult


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

def test_validation_result() -> None:
    res = ValidationResult(metrics={"rmse": 0.1}, is_stable=True)
    assert res.metrics["rmse"] == 0.1
    assert res.is_stable is True

def test_validation_result_empty_metrics() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ValidationResult(metrics={}, is_stable=True)
    assert "Metrics dictionary cannot be empty" in str(excinfo.value)

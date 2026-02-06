from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult


def test_dataset_validation() -> None:
    # 1. File path only - OK
    d2 = Dataset(file_path=Path("data.xyz"))
    assert d2.file_path == Path("data.xyz")

    # 2. No file path - Error (implicit, since field is required)
    with pytest.raises(ValidationError):
        Dataset() # type: ignore[call-arg]

    # 3. Extra fields - Error
    with pytest.raises(ValidationError):
        Dataset(file_path=Path("data.xyz"), extra="forbidden") # type: ignore[call-arg]

def test_structure_metadata_validation() -> None:
    # structure cannot be None
    with pytest.raises(ValidationError):
        StructureMetadata(structure=None) # type: ignore[arg-type]

    # Valid
    atoms = Atoms("H2O")
    meta = StructureMetadata(structure=atoms)
    assert meta.structure == atoms

def test_validation_result_validation() -> None:
    # metrics cannot be empty
    with pytest.raises(ValidationError):
        ValidationResult(metrics={}, is_stable=True)

    # Valid
    res = ValidationResult(metrics={"rmse": 0.1}, is_stable=True)
    assert res.metrics["rmse"] == 0.1

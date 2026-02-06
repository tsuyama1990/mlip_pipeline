from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult


def test_dataset_mutual_exclusivity() -> None:
    # 1. Structures only - OK
    atoms = Atoms("H2O")
    meta = StructureMetadata(structure=atoms)
    d1 = Dataset(structures=[meta])
    assert len(d1.structures) == 1
    assert d1.file_path is None

    # 2. File path only - OK
    d2 = Dataset(file_path=Path("data.xyz"))
    assert len(d2.structures) == 0
    assert d2.file_path == Path("data.xyz")

    # 3. Both - Error
    with pytest.raises(ValidationError) as exc:
        Dataset(structures=[meta], file_path=Path("data.xyz"))
    assert "Dataset cannot have both" in str(exc.value)

    # 4. Neither - OK (empty dataset)
    d4 = Dataset()
    assert len(d4.structures) == 0
    assert d4.file_path is None

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

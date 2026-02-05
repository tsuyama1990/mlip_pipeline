import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.dataset import Dataset
from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult


def test_structure_metadata_valid() -> None:
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(
        structure=atoms,
        source="test",
        generation_method="manual"
    )
    assert meta.structure == atoms
    assert meta.source == "test"

def test_structure_metadata_invalid_structure() -> None:
    with pytest.raises(ValidationError) as excinfo:
        StructureMetadata(
            structure="not an atom", # type: ignore[arg-type]
            source="test",
            generation_method="manual"
        )
    # The validation error message is wrapped by Pydantic
    assert "Input should be an instance of Atoms" in str(excinfo.value)

def test_structure_metadata_extra_fields() -> None:
    atoms = Atoms('H')
    with pytest.raises(ValidationError):
        StructureMetadata(
            structure=atoms,
            source="test",
            generation_method="manual",
            extra_field="fail" # type: ignore[call-arg]
        )

def test_dataset_operations() -> None:
    ds = Dataset(name="test_ds")
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(structure=atoms, source="test", generation_method="manual")

    ds.add(meta)
    assert len(ds) == 1
    assert next(iter(ds)) == meta

def test_validation_result() -> None:
    metric = MetricResult(name="test", passed=True, score=0.99)
    res = ValidationResult(passed=True, metrics=[metric])
    assert res.passed
    assert res.metrics[0].name == "test"

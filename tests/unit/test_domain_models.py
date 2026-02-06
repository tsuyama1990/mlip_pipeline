from pathlib import Path

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models import Dataset, StructureMetadata, ValidationResult


def test_structure_metadata_valid() -> None:
    """
    Tests that a valid StructureMetadata object can be created.
    """
    atoms = Atoms("H2O")
    meta = StructureMetadata(structure=atoms, energy=-10.5, iteration=1)
    assert meta.structure == atoms
    assert meta.energy == -10.5
    assert meta.iteration == 1
    assert meta.forces is None


def test_dataset_initialization() -> None:
    """
    Tests that a Dataset object can be initialized with a file path.
    """
    path = Path("dataset.xyz")
    dataset = Dataset(file_path=path)
    assert dataset.file_path == path


def test_validation_result() -> None:
    """
    Tests that a ValidationResult object can be created with valid metrics.
    """
    res = ValidationResult(metrics={"rmse": 0.1}, is_stable=True)
    assert res.metrics["rmse"] == 0.1
    assert res.is_stable is True


def test_validation_result_empty_metrics() -> None:
    """
    Tests that creating a ValidationResult with empty metrics raises a ValidationError.
    """
    with pytest.raises(ValidationError) as excinfo:
        ValidationResult(metrics={}, is_stable=True)
    assert "Metrics dictionary cannot be empty" in str(excinfo.value)

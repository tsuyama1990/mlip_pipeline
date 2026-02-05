import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structures import StructureMetadata
from mlip_autopipec.domain_models.validation import MetricResult, ValidationResult


def test_structure_metadata_valid() -> None:
    """Test valid StructureMetadata creation."""
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
    metadata = StructureMetadata(
        structure=atoms,
        source="test",
        generation_method="manual"
    )
    assert metadata.source == "test"
    assert len(metadata.structure) == 2

def test_structure_metadata_missing_field() -> None:
    """Test missing required field raises error."""
    atoms = Atoms('H2')
    with pytest.raises(ValidationError):
        StructureMetadata(structure=atoms) # type: ignore

def test_validation_result_valid() -> None:
    """Test valid ValidationResult creation."""
    metric = MetricResult(
        name="test_metric",
        passed=True,
        score=0.99
    )
    result = ValidationResult(
        passed=True,
        metrics=[metric]
    )
    assert result.passed is True
    assert len(result.metrics) == 1
    assert result.metrics[0].name == "test_metric"

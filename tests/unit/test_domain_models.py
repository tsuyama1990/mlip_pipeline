"""Unit tests for domain models."""

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.models import (
    ActiveSet,
    CycleStatus,
    HaltInfo,
    MaterialDNA,
    Potential,
    PotentialType,
    PredictedProperties,
    StructureMetadata,
    StructureStatus,
    Task,
    TaskStatus,
    TaskType,
    UncertaintyState,
)


def test_structure_metadata_creation() -> None:
    """Test creation of StructureMetadata."""
    s = StructureMetadata()
    assert isinstance(s.id, UUID)
    assert s.status == "NEW"
    assert s.features == {}
    assert s.material_dna is None
    assert s.energy is None


def test_structure_metadata_with_features() -> None:
    """Test creation of StructureMetadata with new structured features."""
    dna = MaterialDNA(composition={"Fe": 1.0}, space_group="Im-3m")
    props = PredictedProperties(band_gap=0.0)
    unc = UncertaintyState(gamma_max=5.0)

    s = StructureMetadata(
        material_dna=dna,
        predicted_properties=props,
        uncertainty_state=unc,
        energy=-100.0,
        forces=[[0.1, 0.1, 0.1]],
        stress=[0.0] * 6,
    )

    assert s.material_dna
    assert s.material_dna.composition == {"Fe": 1.0}
    assert s.predicted_properties
    assert s.predicted_properties.band_gap == 0.0
    assert s.uncertainty_state
    assert s.uncertainty_state.gamma_max == 5.0
    assert s.energy == -100.0
    assert s.forces
    assert len(s.forces) == 1
    assert s.stress
    assert len(s.stress) == 6


def test_structure_metadata_validation() -> None:
    """Test validation of StructureMetadata status consistency."""
    # Should fail if CALCULATED but missing energy/forces
    with pytest.raises(ValidationError, match="Energy must be present"):
        StructureMetadata(status=StructureStatus.CALCULATED)

    # Should succeed if CALCULATED and has fields
    s = StructureMetadata(status=StructureStatus.CALCULATED, energy=-10.0, forces=[[0.0, 0.0, 0.0]])
    assert s.status == StructureStatus.CALCULATED


def test_potential_validation() -> None:
    """Test Potential validation."""
    with pytest.raises(ValidationError):
        Potential(path="invalid", type="INVALID", version="1.0")  # type: ignore[arg-type]

    p = Potential(
        path=Path("pot.yace"), type=PotentialType.PACE, version="1.0", parameters={"cutoff": 5.0}
    )
    assert p.type == "PACE"
    assert p.parameters["cutoff"] == 5.0


def test_task_creation() -> None:
    """Test Task creation and status update."""
    t = Task(type=TaskType.DFT)
    assert t.status == TaskStatus.PENDING
    assert t.result == {}


def test_cycle_status_enum() -> None:
    """Test CycleStatus enum values."""
    assert CycleStatus.EXPLORATION == "EXPLORATION"
    assert CycleStatus.CONVERGED == "CONVERGED"


def test_active_set_validation() -> None:
    """Test ActiveSet validation."""
    with pytest.raises(ValidationError):
        ActiveSet(structure_ids=[])  # type: ignore[call-arg]

    s1 = StructureMetadata()
    aset = ActiveSet(structure_ids=[s1.id], selection_criteria="random")
    assert len(aset.structure_ids) == 1


def test_halt_info_creation() -> None:
    """Test creation of HaltInfo."""
    # Not halted
    h = HaltInfo(halted=False)
    assert not h.halted
    assert h.step is None
    assert h.max_gamma is None
    assert h.structure is None

    # Halted
    s = StructureMetadata()
    h = HaltInfo(halted=True, step=100, max_gamma=5.5, structure=s)
    assert h.halted
    assert h.step == 100
    assert h.max_gamma == 5.5
    assert h.structure == s


def test_halt_info_validation() -> None:
    """Test validation of HaltInfo."""
    # Halted but missing info
    with pytest.raises(ValidationError, match="Step must be provided"):
        HaltInfo(halted=True)

    with pytest.raises(ValidationError, match="Max gamma must be provided"):
        HaltInfo(halted=True, step=100)

    with pytest.raises(ValidationError, match="Structure must be provided"):
        HaltInfo(halted=True, step=100, max_gamma=5.5)

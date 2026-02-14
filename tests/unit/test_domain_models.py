"""Unit tests for domain models."""

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from pyacemaker.domain_models.models import (
    ActiveSet,
    CycleStatus,
    Potential,
    PotentialType,
    StructureMetadata,
    Task,
    TaskStatus,
    TaskType,
)


def test_structure_metadata_creation() -> None:
    """Test creation of StructureMetadata."""
    s = StructureMetadata()
    assert isinstance(s.id, UUID)
    assert s.status == "NEW"
    assert s.features == {}


def test_potential_validation() -> None:
    """Test Potential validation."""
    with pytest.raises(ValidationError):
        Potential(path="invalid", type="INVALID", version="1.0")  # type: ignore[arg-type]

    p = Potential(path=Path("pot.yace"), type=PotentialType.PACE, version="1.0")
    assert p.type == "PACE"


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

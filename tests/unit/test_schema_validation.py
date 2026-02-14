"""Tests for domain model schema validation."""

import pytest
from ase import Atoms
from pydantic import ValidationError

from pyacemaker.domain_models.common import PotentialType, StructureStatus
from pyacemaker.domain_models.structure import StructureMetadata


def test_strenum_behavior() -> None:
    """Test that StrEnum behaves like a string."""
    assert StructureStatus.NEW == "NEW"
    assert isinstance(StructureStatus.NEW, str)
    assert f"Status is {StructureStatus.NEW}" == "Status is NEW"
    assert PotentialType.PACE == "PACE"

def test_structure_metadata_features_validation() -> None:
    """Test validation of features dictionary."""
    # Valid Atoms object
    s = StructureMetadata(features={"atoms": Atoms("H2O")})
    assert isinstance(s.features["atoms"], Atoms)

    # Valid primitives
    s = StructureMetadata(features={"count": 1, "label": "test"})
    assert s.features["count"] == 1

    # Invalid object type
    class RandomObj:
        pass

    with pytest.raises(ValidationError, match="unsafe type"):
        StructureMetadata(features={"obj": RandomObj()})

    # Object with todict (should pass)
    class DictObj:
        def todict(self) -> dict[str, object]: return {}

    s = StructureMetadata(features={"dict_obj": DictObj()})
    assert isinstance(s.features["dict_obj"], DictObj)

def test_structure_metadata_energy_validation() -> None:
    """Test energy validation."""
    with pytest.raises(ValidationError, match="finite number"):
        StructureMetadata(energy=float("inf"))

    with pytest.raises(ValidationError, match="physically implausible"):
        StructureMetadata(energy=1e9) # Too high

def test_structure_metadata_forces_validation() -> None:
    """Test forces validation."""
    with pytest.raises(ValidationError, match="physically implausible"):
        StructureMetadata(forces=[[1e5, 0, 0]]) # Too high

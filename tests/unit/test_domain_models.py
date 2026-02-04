import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structures import StructureMetadata


def test_structure_metadata_creation() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(atoms=atoms, source="random", generation_method="rattle")
    assert meta.source == "random"
    assert len(meta.atoms) == 2


def test_structure_metadata_missing_fields() -> None:
    atoms = Atoms("H")
    with pytest.raises(ValidationError):
        StructureMetadata(atoms=atoms)  # type: ignore[call-arg]

import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.domain_models.structures import StructureMetadata


def test_structure_metadata_valid() -> None:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    meta = StructureMetadata(structure=atoms, source="initial", generation_method="random")
    assert meta.structure == atoms
    assert meta.source == "initial"
    assert meta.generation_method == "random"


def test_structure_metadata_invalid_structure() -> None:
    with pytest.raises((ValidationError, TypeError)) as exc:
        StructureMetadata(
            structure="not an atoms object", source="initial", generation_method="random"
        )
    assert "structure must be an ase.Atoms object" in str(exc.value)


def test_structure_metadata_invalid_source() -> None:
    atoms = Atoms("H")
    with pytest.raises(ValidationError):
        StructureMetadata(
            structure=atoms,
            source="invalid_source",  # type: ignore
            generation_method="random",
        )


def test_structure_metadata_invalid_generation_method() -> None:
    atoms = Atoms("H")
    with pytest.raises(ValidationError):
        StructureMetadata(
            structure=atoms,
            source="initial",
            generation_method="invalid_method",  # type: ignore
        )

import pytest
from pydantic import ValidationError
from mlip_autopipec.data_models.inference_models import ExtractedStructure
from ase import Atoms

def test_extracted_structure_valid() -> None:
    atoms = Atoms("Al", positions=[[0, 0, 0]])
    structure = ExtractedStructure(
        atoms=atoms,
        origin_uuid="1234-5678",
        origin_index=0,
        mask_radius=4.0
    )
    assert structure.origin_uuid == "1234-5678"
    assert structure.origin_index == 0
    assert structure.mask_radius == 4.0
    assert structure.atoms == atoms

def test_extracted_structure_invalid() -> None:
    # We raise TypeError in the validator, so we should catch that.
    with pytest.raises((ValidationError, TypeError)):
        ExtractedStructure(
            atoms="not_atoms",
            origin_uuid="1234",
            origin_index=0,
            mask_radius=4.0
        )

    with pytest.raises(ValidationError):
        ExtractedStructure(
             atoms=Atoms("H"),
             origin_uuid="1234",
             # missing origin_index
             mask_radius=4.0
        )

def test_extracted_structure_invalid_atoms_type() -> None:
    # Test explicitly ensuring that 'atoms' field only accepts ase.Atoms
    with pytest.raises((ValidationError, TypeError)) as exc:
        ExtractedStructure(
            atoms=123,
            origin_uuid="uuid",
            origin_index=0,
            mask_radius=1.0
        )
    assert "must be an ase.Atoms object" in str(exc.value) or "atoms" in str(exc.value)

def test_extracted_structure_edge_cases() -> None:
    # Negative index
    # We rely on pydantic/python int which allows negative.
    # But logic might not like it. The model itself doesn't constrain it unless we add ge=0
    # Let's assume standard int.

    # Empty UUID
    s = ExtractedStructure(
        atoms=Atoms(),
        origin_uuid="",
        origin_index=0,
        mask_radius=1.0
    )
    assert s.origin_uuid == ""

    # Negative radius? Logic is used for calculation but usually radius > 0.
    # The config has gt=0, but data model might not.
    # Spec doesn't strictly forbid negative radius in data model, but Config does.
    pass

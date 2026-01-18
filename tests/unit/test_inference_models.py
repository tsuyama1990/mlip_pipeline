import pytest
from ase import Atoms
from pydantic import ValidationError

from mlip_autopipec.data_models.inference_models import ExtractedStructure


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
    with pytest.raises(ValidationError):
        ExtractedStructure(
            atoms="not_atoms", # It allows arbitrary types, but let's check missing fields
            origin_uuid="1234",
            # missing origin_index
            mask_radius=4.0
        )

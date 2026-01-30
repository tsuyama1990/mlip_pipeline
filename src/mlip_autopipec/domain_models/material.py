from typing import Optional
from pydantic import BaseModel, ConfigDict


class MaterialProperties(BaseModel):
    """Properties for a specific material system."""
    model_config = ConfigDict(extra="forbid")

    element: str
    crystal_structure: str
    lattice_constant: float


# Simple in-memory database for now
MATERIAL_DB = {
    "Si": MaterialProperties(element="Si", crystal_structure="diamond", lattice_constant=5.43),
    "Al": MaterialProperties(element="Al", crystal_structure="fcc", lattice_constant=4.05),
    "Cu": MaterialProperties(element="Cu", crystal_structure="fcc", lattice_constant=3.61),
    "Ti": MaterialProperties(element="Ti", crystal_structure="hcp", lattice_constant=2.95),
}

def get_material_properties(element: str) -> Optional[MaterialProperties]:
    return MATERIAL_DB.get(element)

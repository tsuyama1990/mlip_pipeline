
import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.structure import Structure


def test_structure_valid_properties() -> None:
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3),
        species=["H"],
        properties={"temperature": 300.0, "converged": True, "tags": ["a", "b"]}
    )
    assert s.properties["temperature"] == 300.0

def test_structure_invalid_property_key() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"],
            properties={123: "value"} # type: ignore
        )
    assert "Property key must be a string" in str(excinfo.value)

def test_structure_invalid_property_value_type() -> None:
    class DummyObj:
        pass

    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"],
            properties={"obj": DummyObj()}
        )
    assert "Property 'obj' has invalid type" in str(excinfo.value)

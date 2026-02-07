import numpy as np
import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import Structure


def test_structure_properties_valid() -> None:
    """Test that valid properties are accepted."""
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3),
        species=["H"],
        properties={
            "energy": -13.6,
            "band_gap": 1.4,
            "forces": [0.1, 0.2, 0.3],  # List allowed
            "stress": np.eye(3)        # Array allowed
        }
    )
    assert s.properties["energy"] == -13.6
    assert isinstance(s.properties["stress"], np.ndarray)

def test_structure_properties_invalid_key() -> None:
    """Test that non-string keys in properties raise ValidationError."""
    with pytest.raises(ValidationError) as excinfo:
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"],
            properties={123: "value"}  # type: ignore
        )
    # Pydantic catches int key for dict[str, Any]
    assert "Input should be a valid string" in str(excinfo.value)

def test_structure_properties_invalid_value() -> None:
    """Test that unsupported types (e.g. dicts/objects) in properties raise ValidationError."""
    class Foo:
        pass

    # Pydantic V2 might propagate the TypeError directly or wrap it.
    # We allow catching both to be robust.
    with pytest.raises((ValidationError, TypeError)) as excinfo:
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"],
            properties={"nested": {"a": 1}}
        )
    assert "Property 'nested' has invalid type" in str(excinfo.value)

    with pytest.raises((ValidationError, TypeError)) as excinfo:
        Structure(
            positions=np.array([[0.0, 0.0, 0.0]]),
            cell=np.eye(3),
            species=["H"],
            properties={"obj": Foo()}
        )
    assert "Property 'obj' has invalid type" in str(excinfo.value)

def test_structure_properties_default() -> None:
    """Test that properties default to empty dict."""
    s = Structure(
        positions=np.array([[0.0, 0.0, 0.0]]),
        cell=np.eye(3),
        species=["H"]
    )
    assert s.properties == {}

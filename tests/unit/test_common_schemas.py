import pytest
from pydantic import ValidationError
from mlip_autopipec.config.schemas.common import Composition

def test_composition_validation():
    # Valid composition
    comp = Composition({"Fe": 0.5, "Ni": 0.5})
    assert comp.root == {"Fe": 0.5, "Ni": 0.5}

    # Invalid sum
    with pytest.raises(ValidationError):
        Composition({"Fe": 0.5, "Ni": 0.6})

    # Empty dictionary
    with pytest.raises(ValidationError):
        Composition({})

    # Invalid keys (non-chemical symbols)
    with pytest.raises(ValidationError):
        Composition({"Fake": 1.0})

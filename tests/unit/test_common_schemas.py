import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.common import Composition


def test_composition_sum_validation():
    # Valid case
    Composition({"Fe": 0.5, "Ni": 0.5})

    # Invalid case - Sum != 1.0
    with pytest.raises(ValidationError) as excinfo:
        Composition({"Fe": 0.5, "Ni": 0.4})
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)

    # Invalid case - Invalid Symbol
    with pytest.raises(ValidationError) as excinfo:
        Composition({"Xx": 1.0})
    assert "not a valid chemical symbol" in str(excinfo.value)

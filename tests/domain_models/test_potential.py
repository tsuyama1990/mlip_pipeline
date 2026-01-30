from pathlib import Path
from datetime import datetime
import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.potential import Potential


def test_potential_valid():
    """Test creating a valid Potential object."""
    p = Potential(
        path=Path("potentials/my.yace"),
        elements=["Ti", "O"],
        metadata={"rmse": 0.001}
    )
    assert p.path == Path("potentials/my.yace")
    assert p.format == "ace"
    assert p.elements == ["Ti", "O"]
    assert isinstance(p.creation_date, datetime)
    assert p.metadata["rmse"] == 0.001


def test_potential_invalid_format():
    """Test that invalid format raises ValidationError."""
    with pytest.raises(ValidationError):
        Potential(
            path=Path("p.yace"),
            elements=["Ti"],
            format="invalid" # type: ignore
        )

def test_potential_missing_fields():
    """Test missing required fields."""
    with pytest.raises(ValidationError):
        Potential(path=Path("p.yace"))

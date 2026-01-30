from pathlib import Path
import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.potential import Potential


def test_potential_valid():
    p = Potential(path=Path("test.yace"), elements=["Si", "O"])
    assert p.format == "ace"
    assert p.path == Path("test.yace")
    assert p.elements == ["Si", "O"]


def test_potential_invalid_format():
    with pytest.raises(ValidationError):
        Potential(path=Path("test.yace"), elements=["Si"], format="invalid") # type: ignore


def test_potential_extra_forbid():
    with pytest.raises(ValidationError):
        Potential(path=Path("test.yace"), elements=["Si"], extra_field="bad") # type: ignore

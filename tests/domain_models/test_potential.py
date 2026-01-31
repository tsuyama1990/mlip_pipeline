from pathlib import Path
import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.potential import Potential


def test_potential_valid():
    p = Potential(
        path=Path("/tmp/model.yace"),
        format="ace",
        elements=["Ti", "O"],
        metadata={"rmse": 0.01},
    )
    assert p.path == Path("/tmp/model.yace")
    assert p.format == "ace"
    assert p.elements == ["Ti", "O"]
    assert p.metadata["rmse"] == 0.01


def test_potential_invalid_format():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("/tmp/model.yace"),
            format="invalid",  # type: ignore
            elements=["Ti", "O"],
        )


def test_potential_extra_forbidden():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("/tmp/model.yace"),
            format="ace",
            elements=["Ti", "O"],
            extra_field="fail",  # type: ignore
        )

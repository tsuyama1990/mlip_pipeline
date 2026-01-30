from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.potential import Potential


def test_potential_valid():
    p = Potential(
        path=Path("pot.yace"),
        format="ace",
        elements=["Ti", "O"],
        creation_date=datetime.now(),
        metadata={"rmse_E": 0.001}
    )
    assert p.path == Path("pot.yace")
    assert p.format == "ace"
    assert p.elements == ["Ti", "O"]


def test_potential_invalid_format():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("pot.bad"),
            format="bad_format",  # type: ignore
            elements=["Ti"],
            creation_date=datetime.now(),
            metadata={}
        )


def test_potential_extra_field():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("pot.yace"),
            format="ace",
            elements=["Ti"],
            creation_date=datetime.now(),
            metadata={},
            extra_field="should fail"  # type: ignore
        )

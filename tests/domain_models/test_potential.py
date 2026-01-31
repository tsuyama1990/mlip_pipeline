from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models import Potential


def test_potential_valid():
    p = Potential(
        path=Path("potentials/pot.yace"),
        format="ace",
        elements=["Si", "O"],
        creation_date=datetime.now(),
        metadata={"rmse": 0.01},
    )
    assert p.format == "ace"
    assert p.path == Path("potentials/pot.yace")


def test_potential_invalid_format():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("potentials/pot.yace"),
            format="invalid",  # type: ignore
            elements=["Si"],
            creation_date=datetime.now(),
            metadata={},
        )


def test_potential_extra_fields():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("pot.yace"),
            format="ace",
            elements=["Si"],
            creation_date=datetime.now(),
            metadata={},
            extra_field="fail",  # type: ignore
        )

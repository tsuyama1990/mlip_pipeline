import pytest
from pydantic import ValidationError
from mlip_autopipec.domain_models.potential import Potential
from mlip_autopipec.domain_models.config import PotentialConfig
from pathlib import Path
from datetime import datetime

def test_potential_model_valid():
    pot = Potential(
        path=Path("potential.yace"),
        format="ace",
        elements=["Si", "O"],
        creation_date=datetime.now(),
        metadata={"version": "1.0"}
    )
    assert pot.format == "ace"

def test_potential_model_invalid_format():
    with pytest.raises(ValidationError):
        Potential(
            path=Path("pot.txt"),
            format="invalid_format", # type: ignore
            elements=["Fe"],
            creation_date=datetime.now(),
            metadata={}
        )

def test_potential_config_valid():
    pc = PotentialConfig(
        elements=["Si", "C"],
        cutoff=4.5,
        pair_style="hybrid/overlay"
    )
    assert pc.cutoff == 4.5
    assert pc.pair_style == "hybrid/overlay"

def test_potential_config_invalid_cutoff():
    with pytest.raises(ValidationError):
        PotentialConfig(elements=["Al"], cutoff=-1.0)

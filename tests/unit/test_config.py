from typing import Any

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import SimulationConfig


def test_valid_config() -> None:
    data: Any = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2],
            "pseudopotentials": {"Si": "Si.upf"}
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0,
            "max_generations": 10
        },
        "exploration": {
            "strategy": "md",
            "max_temperature": 500.0,
            "steps": 50
        }
    }
    config = SimulationConfig(**data)
    assert config.project_name == "TestProject"
    assert config.dft.ecutwfc == 40.0
    assert config.training.code == "pacemaker"
    assert config.exploration.strategy == "md"

def test_missing_field() -> None:
    data: Any = {
        "project_name": "TestProject",
        # Missing dft
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with pytest.raises(ValidationError) as exc:
        SimulationConfig(**data)
    assert "dft" in str(exc.value)

def test_invalid_type() -> None:
    data: Any = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": "not_a_float", # Error
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with pytest.raises(ValidationError) as exc:
        SimulationConfig(**data)
    assert "ecutwfc" in str(exc.value)

def test_defaults() -> None:
    data: Any = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    config = SimulationConfig(**data)
    # Check defaults for exploration
    assert config.exploration.strategy == "random"
    assert config.exploration.max_temperature == 1000.0

def test_extra_forbid() -> None:
    data: Any = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        },
        "unknown_section": {} # Error
    }
    with pytest.raises(ValidationError) as exc:
        SimulationConfig(**data)
    assert "Extra inputs are not permitted" in str(exc.value)

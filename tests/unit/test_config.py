import pytest
from pydantic import ValidationError

from mlip_autopipec.config.config_model import (
    SimulationConfig,
)


def test_valid_config():
    """Test that a valid configuration is accepted."""
    config_data = {
        "project_name": "TestProject",
        "dft": {
            "code": "qe",
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0,
            "max_generations": 10
        },
        "exploration": {
            "strategy": "random",
            "max_temperature": 500.0,
            "steps": 100
        }
    }
    config = SimulationConfig(**config_data)
    assert config.project_name == "TestProject"
    assert config.dft.ecutwfc == 40.0
    assert config.training.max_generations == 10


def test_invalid_dft_config():
    """Test that invalid DFT configuration raises ValidationError."""
    config_data = {
        "project_name": "TestProject",
        "dft": {
            "code": "unknown",  # Invalid code
            "ecutwfc": 40.0,
            "kpoints": [2, 2, 2]
        },
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**config_data)
    assert "Input should be 'qe' or 'vasp'" in str(excinfo.value)


def test_missing_required_field():
    """Test that missing required field raises ValidationError."""
    config_data = {
        "project_name": "TestProject",
        # Missing dft
        "training": {
            "code": "pacemaker",
            "cutoff": 5.0
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        SimulationConfig(**config_data)
    assert "Field required" in str(excinfo.value)
    assert "dft" in str(excinfo.value)


def test_default_exploration():
    """Test that exploration config has defaults."""
    config_data = {
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
    config = SimulationConfig(**config_data)
    assert config.exploration.strategy == "random"
    assert config.exploration.max_temperature == 1000.0

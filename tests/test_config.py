import pytest
from pydantic import ValidationError
from pathlib import Path
from mlip_autopipec.config.schemas.common import MinimalConfig, TargetSystem, Composition, SimulationGoal
from mlip_autopipec.config.schemas.resources import Resources
from mlip_autopipec.config.schemas.system import SystemConfig

def test_minimal_config_valid():
    config_dict = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        },
        "simulation_goal": {
            "type": "melt_quench"
        }
    }
    config = MinimalConfig(**config_dict)
    assert config.project_name == "TestProject"
    assert config.target_system.elements == ["Al", "Cu"]
    assert config.resources.dft_code == "quantum_espresso"

def test_minimal_config_invalid_composition():
    config_dict = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Al", "Cu"],
            "composition": {"Al": 0.5, "Cu": 0.4}, # Sums to 0.9
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(**config_dict)
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)

def test_minimal_config_invalid_element():
    config_dict = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Xy", "Cu"], # Invalid element
            "composition": {"Xy": 0.5, "Cu": 0.5},
            "crystal_structure": "fcc"
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        MinimalConfig(**config_dict)
    assert "'Xy' is not a valid chemical symbol" in str(excinfo.value)

def test_system_config_immutability():
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(
            elements=["Al"], composition=Composition({"Al": 1.0}), crystal_structure="fcc"
        ),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=1)
    )
    system_config = SystemConfig(
        minimal=minimal,
        working_dir=Path("/tmp/work"),
        db_path=Path("/tmp/work/db.sqlite"),
        log_path=Path("/tmp/work/log.txt")
    )

    with pytest.raises(ValidationError):
        system_config.working_dir = Path("/tmp/other")

def test_resources_validation():
    with pytest.raises(ValidationError):
        Resources(dft_code="quantum_espresso", parallel_cores=0) # Must be > 0

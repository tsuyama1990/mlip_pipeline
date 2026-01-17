import pytest
from pathlib import Path
from pydantic import ValidationError
from mlip_autopipec.config.models import MinimalConfig, SystemConfig, Resources, TargetSystem

def test_valid_minimal_config_parses_correctly():
    """Tests that a valid minimal configuration is parsed without errors."""
    valid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.5},
            "crystal_structure": "fcc",
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4,
            "gpu_enabled": False
        }
    }
    try:
        MinimalConfig.model_validate(valid_config)
    except ValidationError as e:
        pytest.fail(f"Valid configuration failed to parse: {e}")

def test_minimal_config_invalid_composition():
    """Tests that invalid composition raises ValidationError."""
    invalid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.5, "Ni": 0.4}, # Sums to 0.9
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 4
        }
    }
    with pytest.raises(ValidationError, match="Composition fractions must sum to 1.0"):
        MinimalConfig.model_validate(invalid_config)

def test_minimal_config_invalid_resource():
    """Tests that invalid resource configuration raises ValidationError."""
    invalid_config = {
        "project_name": "Test Project",
        "target_system": {
            "elements": ["Fe"],
            "composition": {"Fe": 1.0},
        },
        "resources": {
            "dft_code": "unknown_code", # Invalid enum
            "parallel_cores": 0 # Invalid gt=0
        }
    }
    with pytest.raises(ValidationError):
        MinimalConfig.model_validate(invalid_config)

def test_system_config_immutability():
    """Verify that SystemConfig is immutable."""
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )

    system_config = SystemConfig(
        minimal=minimal,
        working_dir=Path("/tmp/work"),
        db_path=Path("/tmp/work/db.sqlite"),
        log_path=Path("/tmp/work/system.log")
    )

    with pytest.raises(ValidationError):
        system_config.working_dir = Path("/tmp/other")

def test_system_config_structure():
    """Verify SystemConfig structure."""
    minimal = MinimalConfig(
        project_name="Test",
        target_system=TargetSystem(elements=["Fe"], composition={"Fe": 1.0}),
        resources=Resources(dft_code="quantum_espresso", parallel_cores=4)
    )

    system_config = SystemConfig(
        minimal=minimal,
        working_dir=Path("/tmp/work"),
        db_path=Path("/tmp/work/db.sqlite"),
        log_path=Path("/tmp/work/system.log")
    )

    assert isinstance(system_config.working_dir, Path)
    assert system_config.minimal.project_name == "Test"

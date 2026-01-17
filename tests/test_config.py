import pytest
from pydantic import ValidationError

from mlip_autopipec.config.schemas.common import (
    Composition,
    Resources,
    UserInputConfig,
)


def test_resources_validation():
    # Valid
    r = Resources(dft_code="quantum_espresso", parallel_cores=4, gpu_enabled=True)
    assert r.parallel_cores == 4

    # Invalid cores
    with pytest.raises(ValidationError):
        Resources(dft_code="quantum_espresso", parallel_cores=0)

    # Invalid dft code
    with pytest.raises(ValidationError):
        Resources(dft_code="unknown", parallel_cores=4)

def test_composition_validation():
    # Valid
    c = Composition({"Fe": 0.8, "Ni": 0.2})
    assert c.root["Fe"] == 0.8

    # Invalid Sum
    with pytest.raises(ValidationError) as exc:
        Composition({"Fe": 0.5, "Ni": 0.2})
    assert "sum to 1.0" in str(exc.value)

    # Invalid Symbol
    with pytest.raises(ValidationError) as exc:
        Composition({"Fe": 0.8, "Xx": 0.2})
    assert "valid chemical symbol" in str(exc.value)

def test_user_input_config_full():
    config_dict = {
        "project_name": "TestProject",
        "target_system": {
            "elements": ["Fe", "Ni"],
            "composition": {"Fe": 0.7, "Ni": 0.3},
            "crystal_structure": "fcc"
        },
        "simulation_goal": {
            "type": "melt_quench",
            "temperature_range": [300, 1500]
        },
        "resources": {
            "dft_code": "quantum_espresso",
            "parallel_cores": 16
        }
    }

    config = UserInputConfig(**config_dict)
    assert config.project_name == "TestProject"
    assert config.resources.parallel_cores == 16
    assert config.target_system.elements == ["Fe", "Ni"]

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError) as exc:
        Resources(parallel_cores=4, extra_field="bad")
    assert "Extra inputs are not permitted" in str(exc.value)

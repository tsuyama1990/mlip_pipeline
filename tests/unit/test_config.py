from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.config.models import MLIPConfig, RuntimeConfig, SystemConfig
from mlip_autopipec.config.schemas.common import TargetSystem
from mlip_autopipec.config.schemas.dft import DFTConfig

# --- TargetSystem Tests ---

def test_target_system_valid():
    ts = TargetSystem(
        elements=["Fe", "Ni"],
        composition={"Fe": 0.7, "Ni": 0.3},
        crystal_structure="fcc"
    )
    assert ts.elements == ["Fe", "Ni"]
    assert ts.composition["Fe"] == 0.7

def test_target_system_invalid_element():
    with pytest.raises(ValidationError) as excinfo:
        TargetSystem(
            elements=["Fee", "Ni"],
            composition={"Fe": 0.7, "Ni": 0.3}
        )
    assert "not a valid chemical symbol" in str(excinfo.value)

def test_target_system_invalid_composition_sum():
    with pytest.raises(ValidationError) as excinfo:
        TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.1}
        )
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)

def test_target_system_invalid_composition_sum_3_elements():
    """Test composition sum validation with 3 elements."""
    with pytest.raises(ValidationError) as excinfo:
        TargetSystem(
            elements=["Fe", "Ni", "Al"],
            composition={"Fe": 0.33, "Ni": 0.33, "Al": 0.33} # Sum 0.99
        )
    assert "Composition fractions must sum to 1.0" in str(excinfo.value)

def test_target_system_invalid_composition_symbol():
    """Test composition contains invalid symbol."""
    # This raises ValueError inside the validator, which Pydantic wraps in ValidationError
    with pytest.raises(ValidationError) as excinfo:
        TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 0.5, "Ni": 0.4, "ZZZ": 0.1}
        )
    assert "not a valid chemical symbol" in str(excinfo.value)

def test_target_system_invalid_composition_value_range():
    """Test composition contains value > 1.0 or < 0.0."""
    with pytest.raises(ValidationError) as excinfo:
        TargetSystem(
            elements=["Fe", "Ni"],
            composition={"Fe": 1.1, "Ni": -0.1}
        )
    assert "must be between 0.0 and 1.0" in str(excinfo.value)

# --- DFTConfig Tests ---

def test_dft_config_valid(tmp_path):
    dft = DFTConfig(
        pseudopotential_dir=tmp_path,
        ecutwfc=30.0,
        kspacing=0.15,
        nspin=2
    )
    assert dft.ecutwfc == 30.0

def test_dft_config_invalid_path():
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(
            pseudopotential_dir=Path("/non/existent/path"),
            ecutwfc=30.0,
            kspacing=0.15
        )
    assert "Pseudopotential directory does not exist" in str(excinfo.value)

def test_dft_config_negative_cutoff(tmp_path):
    with pytest.raises(ValidationError) as excinfo:
        DFTConfig(
            pseudopotential_dir=tmp_path,
            ecutwfc=-10.0,
            kspacing=0.15
        )
    assert "greater than 0" in str(excinfo.value)

# --- MLIPConfig Tests ---

def test_mlip_config_full(tmp_path):
    config = MLIPConfig(
        target_system=TargetSystem(
            elements=["Al"],
            composition={"Al": 1.0}
        ),
        dft=DFTConfig(
            pseudopotential_dir=tmp_path,
            ecutwfc=40.0,
            kspacing=0.1
        ),
        runtime=RuntimeConfig(
            work_dir=Path("scratch")
        )
    )
    assert config.dft.ecutwfc == 40.0
    assert config.runtime.database_path == Path("mlip.db")

def test_mlip_config_forbidden_extra(tmp_path):
    """Test that extra fields are forbidden in MLIPConfig."""
    data = {
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0}
        },
        "dft": {
            "pseudopotential_dir": str(tmp_path),
            "ecutwfc": 40.0,
            "kspacing": 0.1,
            "nspin": 1
        },
        "extra_field": "should fail"
    }
    with pytest.raises(ValidationError) as excinfo:
        MLIPConfig.model_validate(data)
    assert "Extra inputs are not permitted" in str(excinfo.value)

def test_mlip_config_missing_field(tmp_path):
    """Test missing required field."""
    data = {
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0}
        }
        # Missing DFT
    }
    with pytest.raises(ValidationError) as excinfo:
        MLIPConfig.model_validate(data)
    assert "Field required" in str(excinfo.value)
    assert "dft" in str(excinfo.value)

def test_mlip_config_nested_validation_error(tmp_path):
    """Test that nested validation errors are propagated."""
    data = {
        "target_system": {
            "elements": ["Al"],
            "composition": {"Al": 1.0}
        },
        "dft": {
            "pseudopotential_dir": str(tmp_path),
            "ecutwfc": -40.0, # Invalid
            "kspacing": 0.1
        }
    }
    with pytest.raises(ValidationError) as excinfo:
        MLIPConfig.model_validate(data)
    assert "greater than 0" in str(excinfo.value)

# --- SystemConfig Tests (Legacy/Strict) ---

def test_system_config_strict_fields():
    """Test that SystemConfig forbids extra fields even with arbitrary_types_allowed."""
    data = {
        "working_dir": "_work",
        "random_extra": "not_allowed"
    }
    with pytest.raises(ValidationError) as excinfo:
        SystemConfig.model_validate(data)
    assert "Extra inputs are not permitted" in str(excinfo.value)

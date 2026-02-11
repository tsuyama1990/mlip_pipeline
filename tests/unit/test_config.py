import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    AdaptiveGeneratorConfig,
    DFTOracleConfig,
    GlobalConfig,
    LAMMPSDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    PacemakerTrainerConfig,
)


def test_valid_config() -> None:
    """Test that a valid configuration is parsed correctly."""
    config_data = {
        "orchestrator": {
            "work_dir": "test_project",
            "n_iterations": 5,
        },
        "generator": {
            "type": "mock",
            "n_candidates": 20,
        },
        "oracle": {
            "type": "mock",
            "noise_std": 0.05,
        },
        "trainer": {
            "type": "mock",
        },
        "dynamics": {
            "type": "mock",
            "steps": 500,
        },
        "validator": {
            "type": "mock",
        },
    }
    config = GlobalConfig.model_validate(config_data)

    assert config.orchestrator.n_iterations == 5
    assert config.orchestrator.work_dir.name == "test_project"
    assert isinstance(config.generator, MockGeneratorConfig)
    assert config.generator.n_candidates == 20
    assert isinstance(config.oracle, MockOracleConfig)
    assert config.oracle.noise_std == 0.05


def test_advanced_config() -> None:
    """Test parsing of advanced component configurations."""
    config_data = {
        "orchestrator": {"work_dir": "adv_project"},
        "generator": {
            "type": "adaptive",
            "md_mc_ratio": 0.8,
            "temperature_max": 2000.0,
        },
        "oracle": {
            "type": "dft",
            "code": "vasp",
            "kspacing": 0.03,
        },
        "trainer": {
            "type": "pacemaker",
            "max_epochs": 500,
        },
        "dynamics": {
            "type": "lammps",
            "steps": 50000,
            "timestep": 0.002,
        },
        "validator": {"type": "mock"},
    }
    config = GlobalConfig.model_validate(config_data)

    assert isinstance(config.generator, AdaptiveGeneratorConfig)
    assert config.generator.md_mc_ratio == 0.8
    assert isinstance(config.oracle, DFTOracleConfig)
    assert config.oracle.code == "vasp"
    assert isinstance(config.trainer, PacemakerTrainerConfig)
    assert config.trainer.max_epochs == 500
    assert isinstance(config.dynamics, LAMMPSDynamicsConfig)
    assert config.dynamics.timestep == 0.002


def test_invalid_iterations() -> None:
    """Test that negative iterations raise ValidationError."""
    config_data = {
        "orchestrator": {
            "work_dir": "test_project",
            "n_iterations": -1,  # Invalid
        },
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig.model_validate(config_data)

    assert "Input should be greater than or equal to 1" in str(excinfo.value)


def test_missing_required_field() -> None:
    """Test that missing required fields raise ValidationError."""
    config_data = {
        "orchestrator": {
            "n_iterations": 5,
            # Missing work_dir
        },
        "generator": {"type": "mock"},
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig.model_validate(config_data)

    assert "Field required" in str(excinfo.value)


def test_unknown_type_discriminator() -> None:
    """Test that unknown type in discriminated union raises ValidationError."""
    config_data = {
        "orchestrator": {"work_dir": "test"},
        "generator": {
            "type": "unknown_generator",  # Invalid type
        },
        "oracle": {"type": "mock"},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "validator": {"type": "mock"},
    }
    with pytest.raises(ValidationError) as excinfo:
        GlobalConfig.model_validate(config_data)

    # Pydantic error for discriminated union mismatch
    assert "Input should be 'mock', 'random' or 'adaptive'" in str(excinfo.value) or "Input tag 'unknown_generator' found using 'type' does not match any of the expected tags" in str(excinfo.value)

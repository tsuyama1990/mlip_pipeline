import pytest
from pydantic import ValidationError
from pathlib import Path
from mlip_autopipec.domain_models.config import (
    Config,
    GeneratorType,
    DynamicsType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def test_valid_config():
    valid_data = {
        "orchestrator": {
            "work_dir": "/tmp/work_dir",
            "max_cycles": 5,
        },
        "generator": {
            "type": GeneratorType.RANDOM,
            "seed": 123,
        },
        "oracle": {
            "type": OracleType.MOCK,
        },
        "trainer": {
            "type": TrainerType.MOCK,
        },
        "dynamics": {
            "type": DynamicsType.MOCK,
        },
        "validator": {
            "type": ValidatorType.MOCK,
        },
    }
    config = Config(**valid_data)
    assert config.orchestrator.work_dir == Path("/tmp/work_dir")
    assert config.generator.type == GeneratorType.RANDOM
    assert config.generator.seed == 123


def test_invalid_config_missing_field():
    invalid_data = {
        "orchestrator": {
            # Missing work_dir
            "max_cycles": 5,
        },
        # Missing other components
    }
    with pytest.raises(ValidationError) as excinfo:
        Config(**invalid_data)
    # Check for missing field error
    assert "Field required" in str(excinfo.value)


def test_invalid_config_extra_field():
    invalid_data = {
        "orchestrator": {
            "work_dir": "/tmp/work_dir",
            "extra_field": "not_allowed",
        },
        "generator": {"type": GeneratorType.RANDOM},
        "oracle": {"type": OracleType.MOCK},
        "trainer": {"type": TrainerType.MOCK},
        "dynamics": {"type": DynamicsType.MOCK},
        "validator": {"type": ValidatorType.MOCK},
    }
    with pytest.raises(ValidationError) as excinfo:
        Config(**invalid_data)
    assert "Extra inputs are not permitted" in str(excinfo.value)


def test_discriminated_union_dynamics():
    # Test LAMMPS config
    lammps_data = {
        "type": DynamicsType.LAMMPS,
        "input_filename": "in.lammps",
        "log_filename": "lammps.log",
        "driver_filename": "driver.py",
        "timestep": 0.002,
    }
    # We need full config to validate via Config or just validate sub-model?
    # Config defines dynamics as Union, so let's try to parse it via Config
    full_data = {
        "orchestrator": {"work_dir": "/tmp"},
        "generator": {"type": GeneratorType.RANDOM},
        "oracle": {"type": OracleType.MOCK},
        "trainer": {"type": TrainerType.MOCK},
        "dynamics": lammps_data,
        "validator": {"type": ValidatorType.MOCK},
    }
    config = Config(**full_data)
    assert config.dynamics.type == DynamicsType.LAMMPS
    assert config.dynamics.timestep == 0.002

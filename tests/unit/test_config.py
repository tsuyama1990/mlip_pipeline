import pytest
from pydantic import ValidationError
from pathlib import Path
from mlip_autopipec.domain_models.config import (
    Config,
    OrchestratorConfig,
    GeneratorType,
    DynamicsType,
    OracleType,
    TrainerType,
    ValidatorType,
    DFTOracleConfig,
    CalculatorType,
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


def test_dft_oracle_command_validation():
    # Valid
    config = DFTOracleConfig(
        type=OracleType.DFT,
        calculator_type=CalculatorType.ESPRESSO,
        command="mpirun -np 4 pw.x"
    )
    assert config.command == "mpirun -np 4 pw.x"

    # Invalid (Shell Injection)
    with pytest.raises(ValidationError) as excinfo:
        DFTOracleConfig(
            type=OracleType.DFT,
            command="rm -rf /; echo hello"
        )
    # The error message changed in the implementation, update the assertion
    assert "forbidden shell operators" in str(excinfo.value)


def test_work_dir_traversal_validation():
    with pytest.raises(ValidationError) as excinfo:
        OrchestratorConfig(work_dir="/tmp/../etc/passwd")
    assert "Path traversal (..) not allowed" in str(excinfo.value)

def test_config_constraints():
    # Test ge, le constraints
    with pytest.raises(ValidationError):
        OrchestratorConfig(work_dir="/tmp", max_cycles=0) # Should be >= 1

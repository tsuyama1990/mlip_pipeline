from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    ExecutionMode,
    GeneratorType,
    OracleType,
)


def test_generator_config_defaults() -> None:
    config = GeneratorConfig()
    assert config.type == GeneratorType.MOCK
    assert config.ratio_ab_initio == 0.1


def test_generator_config_validation() -> None:
    with pytest.raises(ValidationError):
        GeneratorConfig(ratio_ab_initio=-0.1)
    with pytest.raises(ValidationError):
        GeneratorConfig(ratio_ab_initio=1.1)


def test_oracle_config_defaults() -> None:
    config = OracleConfig()
    assert config.type == OracleType.MOCK
    assert config.dft_code is None
    assert config.command is None


def test_orchestrator_config_defaults() -> None:
    config = OrchestratorConfig()
    assert config.max_cycles == 1
    assert config.work_dir == Path("./experiments")
    assert config.execution_mode == ExecutionMode.MOCK


def test_global_config_instantiation() -> None:
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(),
        validator=ValidatorConfig(),
    )
    assert config.orchestrator.max_cycles == 1
    assert config.generator.type == GeneratorType.MOCK

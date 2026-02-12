from pathlib import Path

import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import (
    DynamicsConfig,
    ExplorationPolicyConfig,
    GeneratorConfig,
    GlobalConfig,
    OracleConfig,
    OrchestratorConfig,
    TrainerConfig,
    ValidatorConfig,
)
from mlip_autopipec.domain_models.enums import (
    ActiveSetMethod,
    ExecutionMode,
    GeneratorType,
    OracleType,
)


def test_generator_config_defaults() -> None:
    config = GeneratorConfig()
    assert config.type == GeneratorType.MOCK
    assert config.ratio_ab_initio == 0.1
    # Check new defaults via policy
    assert config.policy.mc_swap_prob == 0.1
    assert config.policy.md_steps == 1000
    assert config.policy.temperature_schedule == [300.0, 600.0, 1200.0]


def test_generator_config_validation() -> None:
    with pytest.raises(ValidationError):
        GeneratorConfig(ratio_ab_initio=-0.1)
    with pytest.raises(ValidationError):
        GeneratorConfig(ratio_ab_initio=1.1)


def test_generator_adaptive_config() -> None:
    policy = ExplorationPolicyConfig(
        mc_swap_prob=0.8,
        md_steps=2000,
        defect_density=0.05,
        strain_range=0.1
    )
    config = GeneratorConfig(policy=policy)
    assert config.policy.mc_swap_prob == 0.8
    assert config.policy.md_steps == 2000
    assert config.policy.defect_density == 0.05
    assert config.policy.strain_range == 0.1


def test_oracle_config_defaults() -> None:
    config = OracleConfig()
    assert config.type == OracleType.MOCK
    assert config.dft_code is None
    assert config.command is None
    # Check new defaults
    assert config.kspacing == 0.04
    assert config.mixing_beta == 0.7


def test_oracle_dft_config() -> None:
    config = OracleConfig(
        kspacing=0.03,
        mixing_beta=0.4
    )
    assert config.kspacing == 0.03
    assert config.mixing_beta == 0.4


def test_trainer_config_defaults() -> None:
    config = TrainerConfig()
    assert config.active_set_method == ActiveSetMethod.NONE
    assert config.selection_ratio == 0.1


def test_dynamics_config_defaults() -> None:
    config = DynamicsConfig()
    assert config.halt_on_uncertainty is True
    assert config.max_gamma_threshold == 5.0


def test_dynamics_halt_config() -> None:
    config = DynamicsConfig(
        halt_on_uncertainty=False,
        max_gamma_threshold=6.0
    )
    assert config.halt_on_uncertainty is False
    assert config.max_gamma_threshold == 6.0


def test_orchestrator_config_defaults() -> None:
    config = OrchestratorConfig(work_dir=Path("./test_experiments"))
    assert config.max_cycles == 1
    assert config.work_dir == Path("./test_experiments")
    assert config.execution_mode == ExecutionMode.MOCK
    assert config.max_candidates == 50


def test_global_config_instantiation() -> None:
    config = GlobalConfig(
        orchestrator=OrchestratorConfig(work_dir=Path("./test_experiments")),
        generator=GeneratorConfig(),
        oracle=OracleConfig(),
        trainer=TrainerConfig(),
        dynamics=DynamicsConfig(),
        validator=ValidatorConfig(),
    )
    assert config.orchestrator.max_cycles == 1
    assert config.generator.type == GeneratorType.MOCK

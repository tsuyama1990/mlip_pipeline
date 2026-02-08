from mlip_autopipec.domain_models.config import (
    ComponentsConfig,
    GlobalConfig,
    MockDynamicsConfig,
    MockGeneratorConfig,
    MockOracleConfig,
    MockValidatorConfig,
    PacemakerTrainerConfig,
    PhysicsBaselineConfig,
)
from mlip_autopipec.domain_models.enums import (
    DynamicsType,
    GeneratorType,
    OracleType,
    TrainerType,
    ValidatorType,
)


def test_pacemaker_trainer_config_defaults() -> None:
    config = PacemakerTrainerConfig(name=TrainerType.PACEMAKER)
    assert config.name == TrainerType.PACEMAKER
    assert config.fitting_weight_energy == 1.0
    assert config.backend_evaluator == "tensorpot"
    assert config.active_set_selection is True
    assert config.active_set_limit == 1000


def test_physics_baseline_config() -> None:
    config = PhysicsBaselineConfig(type="lj", params={"sigma": 3.0, "epsilon": 0.1})
    assert config.type == "lj"
    assert config.params["sigma"] == 3.0


def test_global_config_with_baseline(tmp_path) -> None:
    # Need to construct a valid GlobalConfig
    # This requires creating dummy component configs

    components = ComponentsConfig(
        generator=MockGeneratorConfig(
            name=GeneratorType.MOCK, cell_size=10.0, n_atoms=2, atomic_numbers=[1]
        ),
        oracle=MockOracleConfig(name=OracleType.MOCK),
        trainer=PacemakerTrainerConfig(name=TrainerType.PACEMAKER),
        dynamics=MockDynamicsConfig(name=DynamicsType.MOCK, selection_rate=0.1),
        validator=MockValidatorConfig(name=ValidatorType.MOCK),
    )

    config = GlobalConfig(
        workdir=tmp_path,
        max_cycles=1,
        components=components,
        physics_baseline=PhysicsBaselineConfig(type="zbl", params={}),
    )

    assert config.physics_baseline is not None
    assert config.physics_baseline.type == "zbl"

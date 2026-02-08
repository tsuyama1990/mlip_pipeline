from mlip_autopipec.domain_models.config import (
    GlobalConfig,
    PacemakerTrainerConfig,
    PhysicsBaselineConfig,
)
from mlip_autopipec.domain_models.enums import TrainerType


def test_pacemaker_trainer_config_defaults() -> None:
    config = PacemakerTrainerConfig(name="pacemaker")
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
    from mlip_autopipec.domain_models.config import (
        ComponentsConfig,
        MockDynamicsConfig,
        MockGeneratorConfig,
        MockOracleConfig,
        MockValidatorConfig,
    )

    components = ComponentsConfig(
        generator=MockGeneratorConfig(name="mock", cell_size=10.0, n_atoms=2, atomic_numbers=[1]),
        oracle=MockOracleConfig(name="mock"),
        trainer=PacemakerTrainerConfig(name="pacemaker"),
        dynamics=MockDynamicsConfig(name="mock", selection_rate=0.1),
        validator=MockValidatorConfig(name="mock"),
    )

    config = GlobalConfig(
        workdir=tmp_path,
        max_cycles=1,
        components=components,
        physics_baseline=PhysicsBaselineConfig(type="zbl", params={})
    )

    assert config.physics_baseline is not None
    assert config.physics_baseline.type == "zbl"

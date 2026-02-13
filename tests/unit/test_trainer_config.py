from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.enums import ActiveSetMethod, TrainerType


def test_trainer_config_defaults() -> None:
    config = TrainerConfig(type=TrainerType.MOCK)
    assert config.type == TrainerType.MOCK
    assert config.max_epochs == 100
    assert config.batch_size == 32
    assert config.active_set_method == ActiveSetMethod.NONE
    assert config.selection_ratio == 0.1
    # New defaults
    assert config.seed == 42
    assert config.kappa == 0.01
    assert config.l1_coeffs == 1e-8
    assert config.l2_coeffs == 1e-8


def test_trainer_config_custom() -> None:
    config = TrainerConfig(
        type=TrainerType.PACEMAKER,
        max_epochs=200,
        batch_size=64,
        active_set_method=ActiveSetMethod.MAXVOL,
        selection_ratio=0.5,
        cutoff=6.0,
        order=3,
        basis_size=1000,
        delta_learning="zbl",
        seed=123,
        kappa=0.05,
        l1_coeffs=1e-5,
        l2_coeffs=1e-5
    )
    assert config.type == TrainerType.PACEMAKER
    assert config.max_epochs == 200
    assert config.batch_size == 64
    assert config.active_set_method == ActiveSetMethod.MAXVOL
    assert config.selection_ratio == 0.5
    assert config.cutoff == 6.0
    assert config.order == 3
    assert config.basis_size == 1000
    assert config.delta_learning == "zbl"
    assert config.seed == 123
    assert config.kappa == 0.05
    assert config.l1_coeffs == 1e-5
    assert config.l2_coeffs == 1e-5

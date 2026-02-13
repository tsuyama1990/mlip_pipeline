from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.enums import TrainerType


def test_trainer_config_defaults() -> None:
    config = TrainerConfig(type=TrainerType.PACEMAKER)
    # Using hardcoded expectations as defaults were moved to Pydantic models
    assert config.cutoff == 5.0
    assert config.order == 2
    assert config.basis_size == 500
    assert config.delta_learning is None

def test_trainer_config_custom() -> None:
    config = TrainerConfig(
        type=TrainerType.PACEMAKER,
        cutoff=6.0,
        order=3,
        basis_size=1000,
        delta_learning="zbl"
    )
    assert config.cutoff == 6.0
    assert config.order == 3
    assert config.basis_size == 1000
    assert config.delta_learning == "zbl"

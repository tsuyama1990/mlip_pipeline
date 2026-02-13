from mlip_autopipec.constants import (
    DEFAULT_BASIS_SIZE,
    DEFAULT_CUTOFF,
    DEFAULT_ORDER,
)
from mlip_autopipec.domain_models.config import TrainerConfig
from mlip_autopipec.domain_models.enums import TrainerType


def test_trainer_config_defaults() -> None:
    config = TrainerConfig(type=TrainerType.PACEMAKER)
    assert config.cutoff == DEFAULT_CUTOFF
    assert config.order == DEFAULT_ORDER
    assert config.basis_size == DEFAULT_BASIS_SIZE
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

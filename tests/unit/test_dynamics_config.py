import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import DynamicsConfig
from mlip_autopipec.domain_models.enums import DynamicsType, HybridPotentialType


def test_dynamics_config_defaults():
    config = DynamicsConfig(type=DynamicsType.MOCK)
    assert config.temperature == 300.0
    assert config.steps == 1000
    assert config.timestep == 0.001
    assert config.hybrid_potential is None
    assert config.halt_on_uncertainty is True


def test_dynamics_config_valid():
    config = DynamicsConfig(
        type=DynamicsType.LAMMPS,
        temperature=500.0,
        steps=5000,
        timestep=0.002,
        hybrid_potential=HybridPotentialType.ZBL,
        zbl_cut_inner=0.7,
        zbl_cut_outer=1.5,
    )
    assert config.temperature == 500.0
    assert config.hybrid_potential == HybridPotentialType.ZBL
    assert config.zbl_cut_inner == 0.7


def test_dynamics_config_invalid():
    with pytest.raises(ValidationError):
        DynamicsConfig(type=DynamicsType.MOCK, temperature=-10.0)

    with pytest.raises(ValidationError):
        DynamicsConfig(type=DynamicsType.MOCK, steps=0)

    with pytest.raises(ValidationError):
        DynamicsConfig(type=DynamicsType.MOCK, timestep=0.0)


def test_dynamics_config_hybrid_enum():
    config = DynamicsConfig(type=DynamicsType.MOCK, hybrid_potential="zbl")
    assert config.hybrid_potential == HybridPotentialType.ZBL

    with pytest.raises(ValidationError):
        DynamicsConfig(type=DynamicsType.MOCK, hybrid_potential="invalid")

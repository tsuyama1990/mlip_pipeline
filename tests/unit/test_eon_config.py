import pytest
from pydantic import ValidationError

from mlip_autopipec.domain_models.config import DynamicsConfig, EONConfig
from mlip_autopipec.domain_models.enums import DynamicsType


def test_eon_config_defaults() -> None:
    config = EONConfig()
    assert config.temperature == 300.0
    assert config.prefactor == 1e13
    assert config.search_method == "akmc"
    assert config.client_path == "eonclient"
    assert config.server_script_name == "potential_server.py"
    assert config.potential_format == "yace"


def test_eon_config_validation() -> None:
    # Test valid
    config = EONConfig(temperature=500.0)
    assert config.temperature == 500.0

    # Test invalid temperature
    with pytest.raises(ValidationError):
        EONConfig(temperature=-10.0)

    # Test invalid prefactor
    with pytest.raises(ValidationError):
        EONConfig(prefactor=0.0)

    # Test extra fields forbidden
    with pytest.raises(ValidationError):
        EONConfig(extra_field="invalid") # type: ignore[call-arg]


def test_dynamics_config_with_eon() -> None:
    # Test that we can attach EONConfig to DynamicsConfig
    eon_conf = EONConfig(temperature=100.0)
    dyn_conf = DynamicsConfig(type=DynamicsType.EON, eon=eon_conf)

    assert dyn_conf.type == DynamicsType.EON
    assert dyn_conf.eon is not None
    assert dyn_conf.eon.temperature == 100.0

    # Test defaults (eon should be None by default)
    dyn_conf_default = DynamicsConfig()
    assert dyn_conf_default.eon is None

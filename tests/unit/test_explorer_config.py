from pydantic import ValidationError
import pytest
from mlip_autopipec.config.schemas.exploration import ExplorerConfig

def test_explorer_config_default():
    config = ExplorerConfig()
    assert config.method == "random"

def test_explorer_config_valid():
    config = ExplorerConfig(method="active_learning")
    assert config.method == "active_learning"

def test_explorer_config_invalid():
    with pytest.raises(ValidationError):
        ExplorerConfig(method="magic") # type: ignore

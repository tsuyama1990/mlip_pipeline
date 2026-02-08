import pytest

from mlip_autopipec.components.dynamics import MockDynamics
from mlip_autopipec.components.generator import MockGenerator
from mlip_autopipec.components.oracle import MockOracle
from mlip_autopipec.components.trainer import MockTrainer
from mlip_autopipec.components.validator import MockValidator
from mlip_autopipec.domain_models import ComponentConfig
from mlip_autopipec.factory import create_component


def test_create_generator() -> None:
    """Test creating a MockGenerator."""
    config = ComponentConfig(type="mock")
    component = create_component("generator", config)
    assert isinstance(component, MockGenerator)


def test_create_oracle() -> None:
    """Test creating a MockOracle."""
    config = ComponentConfig(type="mock")
    component = create_component("oracle", config)
    assert isinstance(component, MockOracle)


def test_create_trainer() -> None:
    """Test creating a MockTrainer."""
    config = ComponentConfig(type="mock")
    component = create_component("trainer", config)
    assert isinstance(component, MockTrainer)


def test_create_dynamics() -> None:
    """Test creating a MockDynamics."""
    config = ComponentConfig(type="mock")
    component = create_component("dynamics", config)
    assert isinstance(component, MockDynamics)


def test_create_validator() -> None:
    """Test creating a MockValidator."""
    config = ComponentConfig(type="mock")
    component = create_component("validator", config)
    assert isinstance(component, MockValidator)


def test_unknown_component_type() -> None:
    """Test raising ValueError for unknown component type."""
    config = ComponentConfig(type="unknown_magic")
    with pytest.raises(ValueError, match="Unknown generator type: unknown_magic"):
        create_component("generator", config)


def test_unknown_component_category() -> None:
    """Test raising ValueError for unknown component category."""
    config = ComponentConfig(type="mock")
    with pytest.raises(ValueError, match="Unknown component category: unknown_category"):
        create_component("unknown_category", config)

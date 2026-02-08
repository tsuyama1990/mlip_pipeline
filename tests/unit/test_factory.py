import sys
from pathlib import Path

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
    with pytest.raises(ValueError, match="Unknown unknown_category type: mock"):
        create_component("unknown_category", config)


def test_dynamic_loading(tmp_path: Path) -> None:
    """Test loading a component from a python path."""
    # Create a dummy module
    module_path = tmp_path / "custom_component.py"
    module_path.write_text("""
class CustomGenerator:
    def __init__(self, **kwargs):
        pass
    def generate(self, potential=None):
        return []
""")

    # Add tmp_path to sys.path so we can import it
    sys.path.insert(0, str(tmp_path))
    try:
        config = ComponentConfig(type="custom_component.CustomGenerator")
        component = create_component("generator", config)
        assert component.__class__.__name__ == "CustomGenerator"
    finally:
        sys.path.pop(0)


def test_dynamic_loading_fail() -> None:
    """Test failure when loading invalid path."""
    config = ComponentConfig(type="non_existent.Module.Class")
    # This falls through to "Unknown {category} type" because _load_class_from_path raises ValueError internally
    # which is caught in create_component (Wait, I catch ValueError in create_component?)
    # My implementation catches ValueError and passes.
    # So it raises "Unknown generator type: ..."
    with pytest.raises(ValueError, match="Unknown generator type"):
        create_component("generator", config)

import contextlib
import importlib
import logging
from typing import Any, cast

from mlip_autopipec.components.dynamics import MockDynamics
from mlip_autopipec.components.generator import MockGenerator
from mlip_autopipec.components.oracle import MockOracle
from mlip_autopipec.components.trainer import MockTrainer
from mlip_autopipec.components.validator import MockValidator
from mlip_autopipec.domain_models import ComponentConfig

logger = logging.getLogger(__name__)

# Default registry with core components
# This can be extended at runtime
REGISTRY: dict[str, dict[str, type[Any]]] = {
    "generator": {"mock": MockGenerator},
    "oracle": {"mock": MockOracle},
    "trainer": {"mock": MockTrainer},
    "dynamics": {"mock": MockDynamics},
    "validator": {"mock": MockValidator},
}


def register_component(category: str, type_name: str, component_class: type[Any]) -> None:
    """
    Registers a new component type.
    """
    if category not in REGISTRY:
        REGISTRY[category] = {}
    REGISTRY[category][type_name] = component_class
    logger.debug(f"Registered {category} component: {type_name} -> {component_class}")


def _load_class_from_path(path: str) -> type[Any]:
    """
    Loads a class from a dot-separated path string (e.g., 'pkg.module.ClassName').
    """
    try:
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return cast(type[Any], getattr(module, class_name))
    except (ImportError, AttributeError, ValueError) as e:
        msg = f"Failed to load component from path '{path}': {e}"
        raise ValueError(msg) from e


def create_component(category: str, config: ComponentConfig) -> Any:
    """
    Instantiates a component based on category and configuration.

    Supports:
    1. Registered aliases (e.g., type="mock")
    2. Full python paths (e.g., type="my_pkg.MyClass")
    """
    # 1. Check registry
    component_class = REGISTRY.get(category, {}).get(config.type)

    # 2. If not found, try loading as path
    if not component_class and "." in config.type:
        with contextlib.suppress(ValueError):
            component_class = _load_class_from_path(config.type)

    if not component_class:
        msg = f"Unknown {category} type: {config.type}"
        raise ValueError(msg)

    # Extract extra arguments from config
    kwargs = config.model_dump(exclude={"type"})

    # Instantiate
    try:
        return component_class(**kwargs)
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
             # This might happen if config has extra fields not supported by component
             # We could log a warning, but for strictness we re-raise
             pass
        raise

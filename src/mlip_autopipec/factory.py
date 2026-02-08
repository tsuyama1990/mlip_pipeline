from typing import Any

from mlip_autopipec.components.dynamics import MockDynamics
from mlip_autopipec.components.generator import MockGenerator
from mlip_autopipec.components.oracle import MockOracle
from mlip_autopipec.components.trainer import MockTrainer
from mlip_autopipec.components.validator import MockValidator
from mlip_autopipec.domain_models import ComponentConfig

GENERATORS: dict[str, type[Any]] = {"mock": MockGenerator}
ORACLES: dict[str, type[Any]] = {"mock": MockOracle}
TRAINERS: dict[str, type[Any]] = {"mock": MockTrainer}
DYNAMICS: dict[str, type[Any]] = {"mock": MockDynamics}
VALIDATORS: dict[str, type[Any]] = {"mock": MockValidator}

CATEGORIES = {
    "generator": GENERATORS,
    "oracle": ORACLES,
    "trainer": TRAINERS,
    "dynamics": DYNAMICS,
    "validator": VALIDATORS,
}


def create_component(category: str, config: ComponentConfig) -> Any:
    """
    Instantiates a component based on category and configuration.
    """
    category_map = CATEGORIES.get(category)
    if not category_map:
        msg = f"Unknown component category: {category}"
        raise ValueError(msg)

    component_class = category_map.get(config.type)
    if not component_class:
        msg = f"Unknown {category} type: {config.type}"
        raise ValueError(msg)

    # Extract extra arguments from config
    kwargs = config.model_dump(exclude={"type"})

    # Instantiate
    try:
        return component_class(**kwargs)
    except TypeError as e:
        # Fallback: if initialization fails due to arguments, try without?
        # No, better to fail loud or fix components.
        # But for robustness with Mocks that might not have __init__, we can check.
        if not kwargs:
            raise  # If no kwargs, it's a different error
        # Check if it's an "unexpected keyword argument" error
        if "unexpected keyword argument" in str(e):
            # For development convenience, we might want to warn and ignore?
            # But SPEC says "strict".
            # I will assume components must handle their config.
            pass
        raise

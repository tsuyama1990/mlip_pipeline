from typing import Any

from mlip_autopipec.domain_models import (
    MockDynamicsConfig,
    MockOracleConfig,
    MockTrainerConfig,
)
from mlip_autopipec.infrastructure.mocks import MockDynamics, MockOracle, MockTrainer


def create_component(config: Any) -> Any:
    """
    Factory function to create components based on configuration.

    Args:
        config: A configuration object (Pydantic model).

    Returns:
        The instantiated component.

    Raises:
        ValueError: If the configuration type is unknown.
    """
    if isinstance(config, MockOracleConfig):
        return MockOracle(config)
    if isinstance(config, MockTrainerConfig):
        return MockTrainer(config)
    if isinstance(config, MockDynamicsConfig):
        return MockDynamics(config)

    msg = f"Unknown config type: {type(config)}"
    raise ValueError(msg)

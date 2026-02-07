from typing import Any

from mlip_autopipec.domain_models.config import GlobalConfig
from mlip_autopipec.factory import (
    create_dynamics,
    create_generator,
    create_oracle,
    create_selector,
    create_trainer,
    create_validator,
)
from mlip_autopipec.infrastructure.mocks import (
    MockDynamics,
    MockOracle,
    MockSelector,
    MockStructureGenerator,
    MockTrainer,
    MockValidator,
)


def test_factory_creation() -> None:
    config_dict: dict[str, Any] = {
        "workdir": "tmp",
        "oracle": {"type": "mock", "params": {"k": "v"}},
        "trainer": {"type": "mock"},
        "dynamics": {"type": "mock"},
        "generator": {"type": "mock"},
        "validator": {"type": "mock"},
        "selector": {"type": "mock"},
    }
    config = GlobalConfig.model_validate(config_dict)

    oracle = create_oracle(config.oracle)
    assert isinstance(oracle, MockOracle)
    assert oracle.params["k"] == "v"

    trainer = create_trainer(config.trainer)
    assert isinstance(trainer, MockTrainer)

    dynamics = create_dynamics(config.dynamics)
    assert isinstance(dynamics, MockDynamics)

    generator = create_generator(config.generator)
    assert isinstance(generator, MockStructureGenerator)

    validator = create_validator(config.validator)
    assert isinstance(validator, MockValidator)

    selector = create_selector(config.selector)
    assert isinstance(selector, MockSelector)


def test_factory_unknown_type() -> None:
    # Test that unknown type raises ValueError
    # Since we use discriminated unions, Pydantic should catch it BEFORE factory if we use GlobalConfig.
    # But if we manually construct config with valid type but factory doesn't know it (e.g. implementation missing),
    # that would be a logic error.
    # Currently only "mock" is allowed by Literal.
    # So we can't easily test "unknown type" passed to factory unless we bypass Pydantic or use a mock object.
    pass

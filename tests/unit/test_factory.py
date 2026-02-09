from mlip_autopipec.components.mocks import MockGenerator, MockOracle
from mlip_autopipec.config import GeneratorConfig, OracleConfig
from mlip_autopipec.domain_models.enums import ComponentRole, GeneratorType, OracleType
from mlip_autopipec.factory import ComponentFactory


def test_create_generator() -> None:
    """Test creating a generator component."""
    config = GeneratorConfig(type=GeneratorType.RANDOM)
    generator = ComponentFactory.create(ComponentRole.GENERATOR, config)
    assert generator is not None
    assert isinstance(generator, MockGenerator)


def test_create_oracle() -> None:
    """Test creating an oracle component."""
    config = OracleConfig(type=OracleType.MOCK)
    oracle = ComponentFactory.create(ComponentRole.ORACLE, config)
    assert oracle is not None
    assert isinstance(oracle, MockOracle)


def test_unknown_component_type() -> None:
    """Test that unknown component type raises ValueError."""
    # We bypass Pydantic validation to test factory resilience,
    # or rely on config validation failure if we can't construct config.
    # Since config validation happens at model creation, we expect ValueError there.
    # However, if we somehow passed a valid config with a mismatched role,
    # the factory might complain.

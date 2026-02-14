"""Tests for adaptive exploration policy."""

from unittest.mock import MagicMock

import pytest

from pyacemaker.core.config import PYACEMAKERConfig, StructureGeneratorConfig
from pyacemaker.generator.policy import AdaptivePolicy, ExplorationContext
from pyacemaker.generator.strategies import (
    M3GNetStrategy,
    RandomStrategy,
)


@pytest.fixture
def mock_config() -> PYACEMAKERConfig:
    """Mock configuration."""
    config = MagicMock(spec=PYACEMAKERConfig)
    config.structure_generator = MagicMock(spec=StructureGeneratorConfig)
    config.structure_generator.initial_exploration = "m3gnet"
    config.structure_generator.strain_range = 0.1
    config.structure_generator.rattle_amplitude = 0.1
    config.structure_generator.defect_density = 0.01
    return config


def test_adaptive_policy_cold_start(mock_config: MagicMock) -> None:
    """Test policy selects M3GNet for cold start (cycle 0)."""
    policy = AdaptivePolicy(mock_config)
    context = ExplorationContext(cycle=0)

    strategy = policy.decide_strategy(context)

    assert isinstance(strategy, M3GNetStrategy)


def test_adaptive_policy_random(mock_config: MagicMock) -> None:
    """Test policy selects RandomStrategy for normal cycles (cycle > 0)."""
    policy = AdaptivePolicy(mock_config)
    context = ExplorationContext(cycle=1)

    strategy = policy.decide_strategy(context)

    # By default without special context, it should likely return Random
    assert isinstance(strategy, RandomStrategy)


def test_adaptive_policy_high_uncertainty(mock_config: MagicMock) -> None:
    """Test policy selects cautious strategy (Random with small perturbation) for high uncertainty."""
    policy = AdaptivePolicy(mock_config)
    # Mock uncertainty state
    uncertainty = MagicMock()
    uncertainty.gamma_max = 10.0  # High uncertainty
    context = ExplorationContext(cycle=1, uncertainty_state=uncertainty)

    # We expect a cautious strategy, which might be RandomStrategy with small params
    strategy = policy.decide_strategy(context)

    assert isinstance(strategy, RandomStrategy)
    # Check if parameters are "cautious" (e.g. smaller strain/rattle)
    # 0.5 * 0.1 = 0.05
    assert strategy.strain_range == 0.05
    assert strategy.rattle_amplitude == 0.05


def test_policy_consistent_behavior(mock_config: MagicMock) -> None:
    """Test policy produces consistent decisions for the same input."""
    policy = AdaptivePolicy(mock_config)

    # Context with uncertainty
    uncertainty = MagicMock()
    uncertainty.gamma_max = 6.0
    context = ExplorationContext(cycle=1, uncertainty_state=uncertainty)

    decision1 = policy.decide_strategy(context)
    decision2 = policy.decide_strategy(context)

    # Check same strategy type
    assert type(decision1) is type(decision2)

    # Check same internal parameters
    if isinstance(decision1, RandomStrategy) and isinstance(decision2, RandomStrategy):
         assert decision1.strain_range == decision2.strain_range
         assert decision1.rattle_amplitude == decision2.rattle_amplitude

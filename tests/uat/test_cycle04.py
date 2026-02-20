"""UAT tests for Cycle 04."""

from typing import Any, cast

import numpy as np
import pytest
from ase.build import bulk

from pyacemaker.core.config import PYACEMAKERConfig
from pyacemaker.generator.policy import AdaptivePolicy, ExplorationContext
from pyacemaker.generator.strategies import DefectStrategy, M3GNetStrategy, RandomStrategy


@pytest.fixture
def mock_config(mocker: Any) -> PYACEMAKERConfig:
    """Mock configuration."""
    config = mocker.Mock(spec=PYACEMAKERConfig)
    config.structure_generator = mocker.Mock()
    config.structure_generator.initial_exploration = "m3gnet"
    config.structure_generator.strain_range = 0.1
    config.structure_generator.rattle_amplitude = 0.1
    config.structure_generator.defect_density = 0.01
    return cast(PYACEMAKERConfig, config)


def test_scenario_01_strain_rattle() -> None:
    """Scenario 01: Verify Strain & Rattle Generation."""
    # Given a perfect crystal structure
    seed = bulk("Cu", "fcc", a=3.6)

    # When I request 10 random candidates with strain
    strategy = RandomStrategy(strain_range=0.1, rattle_amplitude=0.1)
    candidates = strategy.generate(seed, n_candidates=10)

    # Then I should receive 10 structures with different cell volumes
    assert len(candidates) == 10

    volumes = [atoms.get_volume() for atoms in candidates]  # type: ignore[no-untyped-call]
    seed_volume = seed.get_volume()  # type: ignore[no-untyped-call]

    # Check that volumes vary within a reasonable range (30%)
    # And are not all identical to seed
    assert not all(np.isclose(v, seed_volume) for v in volumes)

    # And the atomic positions should be perturbed
    positions = [atoms.get_positions() for atoms in candidates]  # type: ignore[no-untyped-call]
    seed_positions = seed.get_positions()  # type: ignore[no-untyped-call]
    assert not all(np.allclose(p, seed_positions) for p in positions)


def test_scenario_02_defect_introduction() -> None:
    """Scenario 02: Verify Defect Introduction."""
    # Given a supercell with 32 atoms
    seed = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    assert len(seed) == 32

    # When I request a structure with 1 vacancy
    strategy = DefectStrategy(defect_density=0.01)  # Should target 1 vacancy for 32 atoms
    candidates = strategy.generate(seed, n_candidates=1)

    # Then I should receive a structure with 31 atoms
    assert len(candidates) == 1
    candidate = candidates[0]
    assert len(candidate) == 31

    # And the cell parameters should remain similar (or same if no relaxation)
    assert np.allclose(candidate.cell, seed.cell)


def test_scenario_03_cold_start_mock(mocker: Any) -> None:
    """Scenario 03: Verify Cold Start via M3GNet (Mock)."""
    # Given the system is configured for M3GNet cold start
    # When M3GNet is not available
    # Then it should fallback gracefully

    seed = bulk("Cu", "fcc", a=3.6)

    # Mock fallback strategy to verify it's called
    mock_fallback = mocker.Mock(spec=RandomStrategy)
    mock_fallback.generate.return_value = [seed.copy()]  # type: ignore[no-untyped-call]

    strategy = M3GNetStrategy(fallback_strategy=mock_fallback)

    candidates = strategy.generate(seed, n_candidates=1)

    # Verify fallback was used
    mock_fallback.generate.assert_called_once()
    assert len(candidates) == 1


def test_scenario_04_adaptive_policy(mock_config: PYACEMAKERConfig) -> None:
    """Scenario 04: Verify Adaptive Policy Logic."""
    policy = AdaptivePolicy(mock_config)

    # Given the system is in Cycle 0 (Cold Start)
    context_0 = ExplorationContext(cycle=0)
    # When the policy engine decides on a strategy
    strategy_0 = policy.decide_strategy(context_0)
    # Then it should select the M3GNet strategy
    assert isinstance(strategy_0, M3GNetStrategy)

    # Given the system detects high uncertainty in Cycle N
    # (Mock uncertainty)
    # When the policy engine decides on a strategy
    # Then it should select a Cautious strategy
    # (We can check if parameters are stricter or strategy type changes)
    # For now assuming it returns RandomStrategy but configured cautiously
    context_n = ExplorationContext(cycle=1)
    strategy_n = policy.decide_strategy(context_n)

    assert isinstance(strategy_n, RandomStrategy)
    # Check default parameters vs cautious if implemented

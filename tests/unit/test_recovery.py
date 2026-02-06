from typing import Any

import pytest

from mlip_autopipec.infrastructure.espresso.recovery import RecoveryStrategy


def test_recovery_strategy_default() -> None:
    strategy = RecoveryStrategy()
    # Default strategy should have some built-in recipes
    recipes = strategy.recipes
    assert len(recipes) > 0

    # First recipe check (assuming default exists)
    # We don't check exact values as defaults might change, but it should return a dict
    next_params = strategy.suggest_next_params(1, {})
    assert isinstance(next_params, dict)

def test_recovery_strategy_custom_recipes() -> None:
    custom_recipes: list[dict[str, Any]] = [
        {"mixing_beta": 0.5},
        {"mixing_beta": 0.3, "smearing": "fermi-dirac"},
    ]
    strategy = RecoveryStrategy(recipes=custom_recipes)

    # Attempt 1 -> Index 0
    next_params = strategy.suggest_next_params(1, {})
    assert next_params["mixing_beta"] == 0.5

    # Attempt 2 -> Index 1
    next_params = strategy.suggest_next_params(2, {})
    assert next_params["mixing_beta"] == 0.3
    assert next_params["smearing"] == "fermi-dirac"

def test_recovery_strategy_exhausted() -> None:
    custom_recipes: list[dict[str, Any]] = [{"mixing_beta": 0.5}]
    strategy = RecoveryStrategy(recipes=custom_recipes)

    # Attempt 2 (out of bounds) should raise StopIteration or similar
    with pytest.raises(StopIteration):
        strategy.suggest_next_params(2, {})

def test_recovery_strategy_merge() -> None:
    # Ensure suggested params are merged with current params (override)
    custom_recipes: list[dict[str, Any]] = [{"mixing_beta": 0.1}]
    strategy = RecoveryStrategy(recipes=custom_recipes)

    current = {"ecutwfc": 50, "mixing_beta": 0.7}
    next_params = strategy.suggest_next_params(1, current)

    assert next_params["ecutwfc"] == 50
    assert next_params["mixing_beta"] == 0.1

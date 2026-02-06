import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


class RecoveryStrategy:
    """
    Defines a sequence of parameter overrides to attempt when SCF convergence fails.
    """

    def __init__(self, initial_params: dict[str, Any]) -> None:
        self.initial_params = initial_params
        # Store recipes as partial updates to avoid duplicating the full config in memory
        self._recipes: list[dict[str, Any]] = [
            # Attempt 1: Reduce mixing beta
            {"mixing_beta": 0.3},
            # Attempt 2: Reduce mixing beta more
            {"mixing_beta": 0.1},
            # Attempt 3: Increase smearing (if using smearing) or add it
            {"mixing_beta": 0.1, "smearing": "methfessel-paxton", "sigma": 0.1},
            # Attempt 4: Cold smearing
            {"mixing_beta": 0.1, "smearing": "cold", "sigma": 0.1},
        ]

    def iter_attempts(self) -> Iterator[dict[str, Any]]:
        """
        Yields parameter dictionaries for each recovery attempt.
        The first yield is the initial configuration.
        """
        # First attempt: Original parameters
        yield self.initial_params

        # Recovery attempts
        for i, recipe in enumerate(self._recipes, 1):
            logger.info(f"Recovery attempt {i}: applying overrides {recipe}")
            # Create a new dictionary only when needed, merging initial with recipe
            # This is cleaner than deepcopying upfront
            new_params = self.initial_params.copy()
            new_params.update(recipe)
            yield new_params

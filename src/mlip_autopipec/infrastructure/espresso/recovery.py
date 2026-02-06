from typing import Any


class RecoveryStrategy:
    """
    Provides parameter overrides for SCF convergence recovery.
    """

    def __init__(self) -> None:
        # List of parameter overrides to apply sequentially
        self._recipes: list[dict[str, Any]] = [
            {"mixing_beta": 0.3},
            {"smearing": "methfessel-paxton", "sigma": 0.2},
            {"mixing_beta": 0.1, "electron_maxstep": 200},
        ]

    def get_recipe(self, retry_count: int) -> dict[str, Any] | None:
        """
        Returns the parameter updates for the given retry count (0-based).
        e.g. retry_count=0 returns the first recovery recipe.
        Returns None if retries exhausted.
        """
        if retry_count < 0 or retry_count >= len(self._recipes):
            return None
        return self._recipes[retry_count]

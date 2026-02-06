import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class RecoveryStrategy:
    """
    Manages recovery strategies for failed calculations.
    It provides a sequence of parameter overrides (recipes) to try.
    """

    DEFAULT_RECIPES: ClassVar[list[dict[str, Any]]] = [
        {"mixing_beta": 0.3},
        {"mixing_beta": 0.1, "diagonalization": "cg"},
        {
            "mixing_beta": 0.1,
            "diagonalization": "cg",
            "smearing": "fermi-dirac",
            "mixing_mode": "local-TF",
        },
    ]

    def __init__(self, recipes: list[dict[str, Any]] | None = None) -> None:
        self.recipes = recipes if recipes else self.DEFAULT_RECIPES

        # Validate recipes structure
        if not isinstance(self.recipes, list):
            msg = "Recovery recipes must be a list of dictionaries."
            raise TypeError(msg)

        for i, recipe in enumerate(self.recipes):
            if not isinstance(recipe, dict):
                msg = f"Recovery recipe at index {i} must be a dictionary, got {type(recipe)}."
                raise TypeError(msg)

    def suggest_next_params(
        self, attempt: int, current_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Returns the parameters for the next attempt.
        attempts are 1-based index of recovery attempts (1st retry, 2nd retry...).
        """
        recipe_index = attempt - 1
        if recipe_index < 0 or recipe_index >= len(self.recipes):
            msg = f"No recovery recipe for attempt {attempt}. Max attempts: {len(self.recipes)}"
            raise IndexError(msg)

        recipe = self.recipes[recipe_index]
        logger.info(f"Recovery attempt {attempt}: applying recipe {recipe}")

        # Merge recipe into current params
        new_params = current_params.copy()
        new_params.update(recipe)
        return new_params

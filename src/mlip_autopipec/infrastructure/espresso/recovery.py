import logging
from collections.abc import Callable
from typing import Any, TypeVar

from ase.calculators.calculator import CalculatorError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy:
    """
    Implements a self-healing strategy for DFT calculations.
    Cycles through a list of parameter overrides to attempt convergence.
    """

    def __init__(self, base_params: dict[str, Any]) -> None:
        self.base_params = base_params
        # Define recovery recipes
        self.recipes: list[dict[str, Any]] = [
            {"mixing_beta": 0.3},
            {"mixing_beta": 0.1},
            {"electron_maxstep": 200},  # Increase max steps
            {"mixing_beta": 0.1, "electron_maxstep": 300},
        ]

    def attempt_calculation(self, calculation_func: Callable[[dict[str, Any]], T]) -> T:
        """
        Executes the calculator function with retries.
        calculation_func: A callable that accepts 'params' (dict) and returns T (e.g., Atoms or energy).
        """
        last_error: Exception | None = None

        # First attempt with base params
        try:
            return calculation_func(self.base_params)
        except CalculatorError as e:
            logger.warning(f"Calculation failed with base params: {e}")
            last_error = e

        # Recovery attempts
        for i, recipe in enumerate(self.recipes, 1):
            logger.info(f"Recovery attempt {i}/{len(self.recipes)} with overrides: {recipe}")

            # Merge base params with recipe
            current_params = self.base_params.copy()
            current_params.update(recipe)

            try:
                return calculation_func(current_params)
            except CalculatorError as e:
                logger.warning(f"Recovery attempt {i} failed: {e}")
                last_error = e
            except Exception:
                logger.exception(f"Unexpected error during recovery attempt {i}")
                raise

        # If all failed
        msg = "All recovery attempts failed."
        logger.error(msg)
        if last_error:
            raise last_error
        raise RuntimeError(msg)

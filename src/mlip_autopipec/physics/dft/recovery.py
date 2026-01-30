import logging
from typing import Any

from mlip_autopipec.domain_models.calculation import (
    MemoryError,
    RecoveryConfig,
    SCFError,
    WalltimeError,
)

logger = logging.getLogger("mlip_autopipec.physics.dft.recovery")


class RecoveryHandler:
    """
    Handles recovery strategies for failed DFT calculations using configurable strategies.
    """

    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config

    def apply_fix(
        self, current_params: dict[str, Any], error: Exception, attempt: int
    ) -> dict[str, Any]:
        """
        Return updated parameters based on the error and attempt number.
        attempt is 1-based index (1st retry, 2nd retry...).
        """
        # Identify error type
        strategy_list: list[dict[str, Any]] = []

        if isinstance(error, SCFError):
            strategy_list = self.config.scf_strategies
        elif isinstance(error, MemoryError):
            strategy_list = self.config.memory_strategies
        elif isinstance(error, WalltimeError):
            strategy_list = self.config.walltime_strategies
        else:
            logger.error(f"No recovery strategy for error type: {type(error)}")
            raise ValueError(f"No recovery strategy for error type: {type(error)}")

        # Check if attempt is within bounds
        # attempt 1 corresponds to index 0
        idx = attempt - 1
        if idx >= len(strategy_list):
            logger.error(
                f"No more recovery strategies for {type(error).__name__} (attempt {attempt})"
            )
            raise ValueError(
                f"No more recovery strategies for {type(error).__name__} (attempt {attempt})"
            )

        # Get the fix
        fix = strategy_list[idx]

        logger.info(
            f"Applying recovery strategy for {type(error).__name__} (attempt {attempt}): {fix}"
        )

        # Apply fix to current parameters
        # We create a copy to avoid mutating the original passed dict unexpectedly (though caller handles it)
        new_params = current_params.copy()
        new_params.update(fix)

        return new_params

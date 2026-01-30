from typing import Any

from mlip_autopipec.domain_models.calculation import (
    MemoryError,
    SCFError,
    WalltimeError,
)


class RecoveryHandler:
    """
    Handles recovery strategies for failed DFT calculations.
    """

    def __init__(self) -> None:
        # Define strategies for each error type
        # List of parameter updates to apply sequentially
        self._scf_strategies: list[dict[str, Any]] = [
            {"mixing_beta": 0.3},
            {"mixing_beta": 0.1},
            {"smearing": "mv", "degauss": 0.02},
        ]

        self._memory_strategies: list[dict[str, Any]] = [
            {"diagonalization": "cg"},
            # Maybe reduce mixing_ndim?
            {"mixing_ndim": 4},
        ]

        self._walltime_strategies: list[dict[str, Any]] = [
            {"diagonalization": "cg"},  # Sometimes faster
            # Maybe reduce conv_thr slightly if acceptable? (1e-5 instead of 1e-6)
            {"conv_thr": 1e-5},
        ]

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
            strategy_list = self._scf_strategies
        elif isinstance(error, MemoryError):
            strategy_list = self._memory_strategies
        elif isinstance(error, WalltimeError):
            strategy_list = self._walltime_strategies
        else:
            # Check if it's a generic DFTError that matches one of the above by name?
            # Or fail.
            # Usually we catch specific errors.
            raise ValueError(f"No recovery strategy for error type: {type(error)}")

        # Check if attempt is within bounds
        # attempt 1 corresponds to index 0
        idx = attempt - 1
        if idx >= len(strategy_list):
            raise ValueError(
                f"No more recovery strategies for {type(error).__name__} (attempt {attempt})"
            )

        # Get the fix
        fix = strategy_list[idx]

        # Apply fix to current parameters
        # We create a copy to avoid mutating the original passed dict unexpectedly (though caller handles it)
        new_params = current_params.copy()
        new_params.update(fix)

        return new_params

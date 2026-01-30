
from mlip_autopipec.domain_models.calculation import DFTConfig, DFTError, RecoveryConfig


class RecoveryHandler:
    """Handles error recovery for DFT calculations."""

    def __init__(self, config: RecoveryConfig):
        self.config = config

    def apply_fix(
        self, dft_config: DFTConfig, error: DFTError, attempt: int
    ) -> tuple[DFTConfig, int]:
        """
        Applies a fix to the DFT configuration based on the error.

        Args:
            dft_config: The current DFT configuration.
            error: The error encountered.
            attempt: The current retry attempt number (0-based).

        Returns:
            A tuple containing the new DFT configuration and the incremented attempt number.

        Raises:
            DFTError: If no strategy exists for the error or retries are exhausted.
        """
        error_type = type(error).__name__
        strategies = self.config.strategies.get(error_type)

        if not strategies:
            # No strategy for this error
            raise error

        if attempt >= len(strategies):
            # Exhausted strategies for this error
            raise error

        if attempt >= self.config.max_retries:
            # Global max retries exceeded
            raise error

        # Apply strategy
        strategy = strategies[attempt]
        new_config = dft_config.model_copy(deep=True)

        # Update fields
        for key, value in strategy.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                # Warning? Or allow extra fields if we had them?
                # DFTConfig is extra="forbid", but we added fields like mixing_beta.
                # If strategy tries to set something not in DFTConfig, it fails on assignment if valid?
                # Actually Pydantic models validate on assignment if configured, but setattr bypasses unless validated.
                # But here we are just setting attributes.
                pass

        return new_config, attempt + 1

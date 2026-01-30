from mlip_autopipec.domain_models.calculation import (
    DFTConfig,
    DFTError,
    SCFError,
)


class RecoveryHandler:
    """Handles recovery from DFT calculation failures."""

    MAX_ATTEMPTS = 5

    def apply_fix(self, config: DFTConfig, error: DFTError, attempt: int) -> DFTConfig:
        """
        Apply a fix to the configuration based on the error and attempt number.

        Args:
            config: The failed configuration.
            error: The error encountered.
            attempt: The current retry attempt number (starts at 1).

        Returns:
            A new DFTConfig with modified parameters.

        Raises:
            DFTError: If max attempts reached or no fix available.
        """
        if attempt > self.MAX_ATTEMPTS:
            raise DFTError(f"Max recovery attempts reached ({self.MAX_ATTEMPTS})")

        new_config = config.model_copy()

        if isinstance(error, SCFError):
            # Strategy for SCF convergence
            if attempt == 1:
                # Try softer mixing
                new_config.mixing_beta = 0.3
            elif attempt == 2:
                # Try hotter electrons (Marzari-Vanderbilt)
                new_config.smearing = "mv"
                new_config.degauss = 0.02
            elif attempt == 3:
                # Change diagonalization algorithm
                new_config.diagonalization = "cg"
            else:
                # Further reduce mixing beta
                new_config.mixing_beta = max(0.05, new_config.mixing_beta * 0.8)

            return new_config

        # Add more error handlers here (MemoryError, etc.)

        # If no handler found, re-raise the error
        raise error

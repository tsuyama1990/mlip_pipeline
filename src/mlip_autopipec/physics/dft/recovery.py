
from mlip_autopipec.domain_models.calculation import (
    DFTConfig,
    DFTError,
    SCFError,
    MemoryError,
    WalltimeError,
)


class RecoveryHandler:
    """
    Handles errors in DFT calculations by proposing new configurations.
    """

    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries

    def apply_fix(self, config: DFTConfig, error: DFTError, attempt: int) -> DFTConfig:
        """
        Return a new DFTConfig with modified parameters to resolve the error.
        Raises RuntimeError if max_retries exceeded or no fix found.
        """
        if attempt > self.max_retries:
            raise RuntimeError(f"Max retries ({self.max_retries}) reached. Last error: {error}")

        # Clone config to avoid mutating original
        new_config = config.model_copy()

        if isinstance(error, SCFError):
            return self._handle_scf_error(new_config, attempt)
        elif isinstance(error, WalltimeError):
            return self._handle_walltime_error(new_config, attempt)
        elif isinstance(error, MemoryError):
            # Complex to handle without changing resources
            pass

        # If unknown error or no fix strategy
        raise RuntimeError(f"No recovery strategy for error: {error}")

    def _handle_scf_error(self, config: DFTConfig, attempt: int) -> DFTConfig:
        """
        Strategies for SCF convergence:
        1. Reduce mixing_beta (damping).
        2. Increase smearing (temperature).
        3. Change diagonalization algorithm.
        """
        # Simple hardcoded sequence of fixes based on attempt count relative to this error?
        # Note: 'attempt' passed here is global attempt for the job.

        # Strategy 1: Reduce mixing beta (Attempts 1-2)
        if attempt <= 2:
            config.mixing_beta *= 0.7
            return config

        # Strategy 2: Increase smearing / temperature (Attempts 3)
        if attempt == 3:
            config.degauss = max(config.degauss * 1.5, 0.01)
            # Ensure smearing is set (if it was 'fixed' or something)
            # But default is 'mv'.
            return config

        # Strategy 3: Change diagonalization (Attempt 4)
        if attempt == 4:
            config.diagonalization = "cg" # Conjugate Gradient is more robust but slower than Davidson
            return config

        # Strategy 4: Desperate measures - very small beta, high temp
        config.mixing_beta = 0.1
        config.degauss = 0.05
        return config

    def _handle_walltime_error(self, config: DFTConfig, attempt: int) -> DFTConfig:
        # Increase timeout
        config.timeout = int(config.timeout * 1.5)
        return config

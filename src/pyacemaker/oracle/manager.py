"""DFT Manager module."""

from collections.abc import Iterator

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS, DFTConfig
from pyacemaker.oracle.calculator import create_calculator


class DFTManager:
    """Manages DFT calculations with retry logic."""

    def __init__(self, config: DFTConfig) -> None:
        """Initialize the DFT Manager."""
        self.config = config
        self.logger = logger.bind(name="DFTManager")
        # Pre-compile or store lowercased patterns for efficiency
        self._recoverable_patterns = [p.lower() for p in CONSTANTS.dft_recoverable_errors]

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Check if the error is potentially recoverable via retry (e.g., SCF convergence)."""
        error_msg = str(error).lower()
        return any(pattern in error_msg for pattern in self._recoverable_patterns)

    def compute(self, structure: Atoms) -> Atoms | None:
        """Run a DFT calculation for a single structure with retries.

        Args:
            structure: The atomic structure to calculate.

        Returns:
            The calculated structure with results attached, or None if failed.

        """
        for attempt in range(self.config.max_retries):
            try:
                calc = create_calculator(self.config, attempt)
                structure.calc = calc

                # Log attempt details for debugging
                mixing_beta = calc.parameters["input_data"]["electrons"].get("mixing_beta")
                self.logger.debug(f"DFT Attempt {attempt + 1}: mixing_beta={mixing_beta}")

                # Trigger calculation
                structure.get_potential_energy()  # type: ignore[no-untyped-call]

            except Exception as e:
                # If it's the last attempt, fail
                if attempt == self.config.max_retries - 1:
                    self.logger.warning(f"DFT failed permanently on attempt {attempt + 1}: {e}")
                    break

                # Check if recoverable
                if self._is_recoverable_error(e):
                    self.logger.warning(
                        f"Recoverable DFT error (Attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying..."
                    )
                    continue

                # If not recoverable, stop immediately
                self.logger.exception("Fatal DFT error")
                break

            self.logger.info("DFT calculation successful")
            return structure

        self.logger.error("DFT calculation failed")
        return None

    def compute_batch(self, structures: list[Atoms] | Iterator[Atoms]) -> Iterator[Atoms | None]:
        """Run DFT calculations for a batch of structures (Generator).

        Args:
            structures: List or Iterator of structures.

        Yields:
            Calculated structure or None.

        """
        for s in structures:
            yield self.compute(s)

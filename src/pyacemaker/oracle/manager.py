"""DFT Manager module."""

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import DFTConfig
from pyacemaker.oracle.calculator import create_calculator


class DFTManager:
    """Manages DFT calculations with retry logic."""

    def __init__(self, config: DFTConfig) -> None:
        """Initialize the DFT Manager."""
        self.config = config
        self.logger = logger.bind(name="DFTManager")

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Check if the error is potentially recoverable via retry (e.g., SCF convergence)."""
        error_msg = str(error).lower()
        # Common SCF convergence error messages
        recoverable_patterns = [
            "scf not converged",
            "convergence not achieved",
            "electronic convergence failed",
            # Add more specific patterns as needed
        ]
        return any(pattern in error_msg for pattern in recoverable_patterns)

    def compute(self, structure: Atoms) -> Atoms | None:
        """Run a DFT calculation for a single structure with retries."""
        for attempt in range(self.config.max_retries):
            try:
                # Create a fresh copy for the calculation to avoid side effects on retry?
                # Usually ASE Atoms object is modified in place by calculator.
                # If calculation fails, the calculator state might be bad.
                # We re-assign the calculator.

                calc = create_calculator(self.config, attempt)
                structure.calc = calc

                # Log attempt details for debugging
                mixing_beta = calc.parameters["input_data"]["electrons"].get("mixing_beta")
                self.logger.debug(
                    f"DFT Attempt {attempt + 1}: mixing_beta={mixing_beta}"
                )

                # Trigger calculation
                # get_potential_energy runs the calculation if needed
                energy = structure.get_potential_energy()  # type: ignore[no-untyped-call]

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

            self.logger.info(f"DFT calculation successful (Energy: {energy:.4f})")
            return structure

        self.logger.error("DFT calculation failed")
        return None

    def compute_batch(self, structures: list[Atoms]) -> list[Atoms | None]:
        """Run DFT calculations for a batch of structures."""
        results: list[Atoms | None] = []
        for s in structures:
            # We work on a copy to be safe? Or modify in place?
            # Modify in place is standard for ASE.
            # But we want to return the modified object.
            res = self.compute(s)
            results.append(res)
        return results

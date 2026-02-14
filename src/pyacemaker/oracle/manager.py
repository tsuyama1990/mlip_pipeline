"""DFT Manager module."""

import tempfile
from collections.abc import Iterator

from ase import Atoms
from loguru import logger

from pyacemaker.core.config import CONSTANTS, DFTConfig
from pyacemaker.core.exceptions import DFTError, StructureError
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
        import re
        error_msg = str(error)
        # Use regex search with proper escaping if patterns are literal strings
        return any(re.search(re.escape(pattern), error_msg, re.IGNORECASE) for pattern in self._recoverable_patterns)

    def _apply_periodic_embedding(self, atoms: Atoms) -> None:
        """Apply periodic embedding to non-periodic structures (in-place).

        If the structure is non-periodic (all pbc are False), this method sets a cell
        with the configured buffer and enables PBC.
        """
        if not self.config.embedding_enabled:
            return

        # Check if structure is already periodic
        if any(atoms.pbc):
            return

        # Apply periodic embedding: center atoms with buffer and enable PBC
        # vacuum parameter adds space on each side of the bounding box
        # We refer to 'vacuum' as 'buffer' in our config for clarity
        atoms.center(vacuum=self.config.embedding_buffer)  # type: ignore[no-untyped-call]
        atoms.pbc = True
        self.logger.debug(
            f"Applied periodic embedding with buffer {self.config.embedding_buffer} A"
        )

    def compute(self, structure: Atoms) -> Atoms:
        """Run a DFT calculation for a single structure with retries.

        Args:
            structure: The atomic structure to calculate.

        Returns:
            The calculated structure with results attached.

        Raises:
            StructureError: If structure is invalid or too large.
            DFTError: If calculation fails after retries.

        """
        # Security: Validate structure size
        if len(structure) > CONSTANTS.max_atoms_dft:
            msg = f"Structure too large: {len(structure)} atoms (max {CONSTANTS.max_atoms_dft})"
            self.logger.error(msg)
            raise StructureError(msg)

        # Create a deep copy to avoid modifying the input structure
        calc_structure = structure.copy()  # type: ignore[no-untyped-call]
        if not isinstance(calc_structure, Atoms):
            # Should not happen with ASE, but satisfies mypy if copy() returns Any
            msg = "Failed to copy structure (invalid type)"
            self.logger.error(msg)
            raise StructureError(msg)

        # Validate structure content (species)
        # Whitelist of allowed elements can be in config, but for now we check for emptiness
        if not calc_structure.get_chemical_symbols():  # type: ignore[no-untyped-call]
             msg = "Structure contains no atoms"
             self.logger.error(msg)
             raise StructureError(msg)

        # Apply embedding if needed (modifies structure in-place)
        self._apply_periodic_embedding(calc_structure)

        # Create a unique temporary directory for this calculation to ensure thread safety
        with tempfile.TemporaryDirectory(prefix=CONSTANTS.DFT_TEMP_PREFIX) as tmp_dir:
            for attempt in range(self.config.max_retries):
                try:
                    calc = create_calculator(self.config, attempt, directory=tmp_dir)
                    calc_structure.calc = calc

                    # Log attempt details for debugging
                    mixing_beta = calc.parameters["input_data"]["electrons"].get("mixing_beta")
                    self.logger.debug(f"DFT Attempt {attempt + 1}: mixing_beta={mixing_beta}")

                    # Trigger calculation
                    calc_structure.get_potential_energy()  # type: ignore[no-untyped-call]

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
                else:
                    self.logger.info("DFT calculation successful")
                    return calc_structure

        self.logger.error("DFT calculation failed")
        msg = "DFT calculation failed after maximum retries"
        raise DFTError(msg)

    def compute_batch(self, structures: list[Atoms] | Iterator[Atoms]) -> Iterator[Atoms]:
        """Run DFT calculations for a batch of structures (Generator).

        Args:
            structures: List or Iterator of structures.

        Yields:
            Calculated structure. Skips failed calculations.

        """
        # Ensure we process as an iterator to avoid checking list length or materializing
        iterator = iter(structures)
        for s in iterator:
            try:
                result = self.compute(s)
                yield result
            except (DFTError, StructureError):
                self.logger.exception("Skipping structure due to error")
                continue

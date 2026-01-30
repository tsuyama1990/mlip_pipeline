from pathlib import Path

import numpy as np
from ase.io import read # type: ignore

from mlip_autopipec.domain_models.calculation import (
    DFTError,
    DFTResult,
    MemoryError,
    SCFError,
    WalltimeError,
)


class Parser:
    """Parses Quantum Espresso output files."""

    @staticmethod
    def parse(output_path: Path) -> DFTResult:
        """
        Parse a QE output file.

        Args:
            output_path: Path to the output file (usually pw.out).

        Returns:
            DFTResult object containing energy, forces, and stress.

        Raises:
            DFTError: If the calculation failed or parsing failed.
        """
        if not output_path.exists():
            raise DFTError(f"Output file not found: {output_path}")

        content = output_path.read_text(errors="replace")

        # Error detection
        # Case-insensitive checks might be safer, but QE messages are standard.
        if "convergence not achieved" in content:
            raise SCFError("SCF convergence not achieved")
        if "maximum CPU time exceeded" in content:
            raise WalltimeError("Maximum CPU time exceeded")
        if "error while allocating" in content or "out of memory" in content:
            raise MemoryError("Memory allocation failed")

        # Generic error catch (e.g. "Error in routine cdiaghg (151): problems computing cholesky")
        # But allow "Job Done" to proceed.
        # Sometimes "Error" appears in normal text? Unlikely in QE output unless crash.
        # But let's be safe and rely on ASE parsing failure for unknown errors if we don't catch them.

        try:
            # Parse results using ASE
            # index=-1 for the last step (static calc has only 1 step usually)
            # Use index=-1 to ensure we get a single Atoms object, but ASE read can return list if index is not specified
            # or if format supports multiple. 'espresso-out' usually returns list if index not set?
            # Let's be explicit.
            atoms_obj = read(output_path, index=-1, format='espresso-out') # type: ignore

            # Mypy safety: ensure it's not a list (though index=-1 guarantees single item usually)
            if isinstance(atoms_obj, list):
                 # Should not happen with index=-1
                 atoms = atoms_obj[-1]
            else:
                 atoms = atoms_obj

        except Exception as e:
            raise DFTError(f"Failed to parse output with ASE: {e}")

        try:
            energy = atoms.get_potential_energy() # type: ignore[no-untyped-call]
            forces = atoms.get_forces() # type: ignore[no-untyped-call]
        except Exception as e:
             raise DFTError(f"Failed to extract energy/forces: {e}")

        # Stress handling
        stress = None
        try:
            # Try to get stress. ASE raises PropertyNotImplementedError if not present.
            # voigt=False returns 3x3 matrix
            stress = atoms.get_stress(voigt=False) # type: ignore[no-untyped-call]
        except Exception:
            # Stress might not be computed if tstress=.false. or convergence failed badly (but we checked convergence)
            # SPEC says stress is required: "stress: NDArray... optional" in model but here it's "Optional" in Result.
            # Wait, in model it is Optional.
            pass

        # Ensure stress is 3x3 if present
        if stress is not None:
             if stress.shape == (6,):
                 # Convert Voigt to 3x3
                 # Voigt order in ASE: xx, yy, zz, yz, xz, xy
                 xx, yy, zz, yz, xz, xy = stress
                 stress = np.array([
                     [xx, xy, xz],
                     [xy, yy, yz],
                     [xz, yz, zz]
                 ])
             elif stress.shape != (3, 3):
                 # Should not happen if voigt=False worked, but safe guard
                 raise DFTError(f"Unexpected stress shape: {stress.shape}")

        return DFTResult(
            energy=energy,
            forces=forces,
            stress=stress,
            magmoms=None
        )

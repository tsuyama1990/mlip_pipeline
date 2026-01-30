import logging
from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.calculation import (
    DFTError,
    DFTResult,
    MemoryError,
    SCFError,
    WalltimeError,
)
from mlip_autopipec.domain_models.job import JobStatus

logger = logging.getLogger("mlip_autopipec.physics.dft.parser")


class DFTParser:
    """
    Parser for Quantum Espresso output files.
    """

    @staticmethod
    def parse(
        filepath: Path,
        job_id: str,
        work_dir: Path,
        duration: float,
    ) -> DFTResult:
        """
        Parse the output file and return a DFTResult or raise an appropriate DFTError.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Output file not found: {filepath}")

        # Use streaming/line-by-line parsing for error checking to avoid OOM
        try:
            DFTParser._check_errors_streaming(filepath)
        except Exception as e:
            # Re-raise known DFT Errors, wrap others
            if isinstance(e, DFTError):
                raise e
            raise DFTError(f"Failed to read/check output file: {e}") from e

        # If no explicit error found, try to parse with ASE
        try:
            # format='espresso-out' is standard in ASE
            # index=-1 ensures we get the last frame
            atoms = ase.io.read(filepath, format="espresso-out", index=-1)  # type: ignore[no-untyped-call]
        except Exception as e:
            # ASE failed to parse. Could be empty file, truncated, or format error.
            raise DFTError(f"ASE failed to parse output: {e}") from e

        # Ensure we have a single Atoms object
        if isinstance(atoms, list):
            # This handles cases where index=-1 might not be respected by some readers, though unlikely
            atoms = atoms[-1]

        if not isinstance(atoms, Atoms):
            # Should be impossible if list check passed, but for MyPy
            raise DFTError(f"Unexpected type returned by ASE: {type(atoms)}")

        calc = atoms.calc  # type: ignore
        if calc is None:
            raise DFTError("Parsed atoms object has no calculator attached.")

        try:
            energy = calc.get_potential_energy()  # type: ignore[no-untyped-call]
            forces = calc.get_forces()  # type: ignore[no-untyped-call]
        except Exception as e:
            raise DFTError(f"Failed to extract energy/forces: {e}") from e

        try:
            stress = calc.get_stress()  # type: ignore[no-untyped-call]
            if stress is not None and stress.shape == (6,):
                # Voigt to 3x3
                # [xx, yy, zz, yz, xz, xy]
                s = np.zeros((3, 3))
                s[0, 0] = stress[0]
                s[1, 1] = stress[1]
                s[2, 2] = stress[2]
                s[1, 2] = s[2, 1] = stress[3]
                s[0, 2] = s[2, 0] = stress[4]
                s[0, 1] = s[1, 0] = stress[5]
                stress = s
        except Exception:
            stress = None

        magmoms = None
        try:
            magmoms = atoms.get_magnetic_moments()  # type: ignore[no-untyped-call]
        except Exception:
            pass

        # For log content, read tail only
        log_tail = DFTParser._read_tail(filepath, lines=50)

        return DFTResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            work_dir=work_dir,
            duration_seconds=duration,
            log_content=log_tail,
            energy=float(energy),
            forces=np.array(forces),
            stress=stress,
            magmoms=magmoms,
        )

    @staticmethod
    def _check_errors_streaming(filepath: Path) -> None:
        """
        Check for known error patterns in the output file line by line.
        Raises specific DFTErrors.
        """
        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # SCF Convergence
                if "convergence not achieved" in line:
                    raise SCFError("SCF convergence not achieved.")

                # Walltime
                # "Maximum CPU time exceeded" is typical QE message
                if "Maximum CPU time exceeded" in line or "time limit" in line.lower():
                    raise WalltimeError("Maximum CPU time exceeded.")

                # Memory
                # "cannot allocate" is common in Fortran allocatable arrays
                # "out of memory" might be system dependent but sometimes in logs
                line_lower = line.lower()
                if "cannot allocate" in line_lower or "out of memory" in line_lower:
                    raise MemoryError("Memory allocation failed.")

                # Generic errors
                if "Error in routine" in line:
                    # We might want to capture more context here, but streaming makes it hard.
                    # Just logging it for now, or could raise generic if no specific error found later.
                    # But often "Error in routine" is followed by specific cause.
                    # Let's keep scanning.
                    pass

    @staticmethod
    def _read_tail(filepath: Path, lines: int = 50) -> str:
        """Read the last N lines of a file efficiently."""
        # Simple implementation using deque
        from collections import deque

        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            try:
                return "".join(deque(f, lines))
            except Exception:
                return "Failed to read log tail."

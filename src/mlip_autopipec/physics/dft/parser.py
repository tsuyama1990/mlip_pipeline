from pathlib import Path

import ase.io
import numpy as np
from ase import Atoms

from mlip_autopipec.domain_models.calculation import (
    DFTResult,
    SCFError,
    MemoryError,
    WalltimeError,
    DFTError,
)
from mlip_autopipec.domain_models.job import JobStatus


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

        # Read content for error checking
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise DFTError(f"Failed to read output file: {e}") from e

        # Check for specific errors
        DFTParser._check_errors(content)

        # If no explicit error found, try to parse with ASE
        try:
            # format='espresso-out' is standard in ASE
            # index=-1 ensures we get the last frame
            atoms = ase.io.read(filepath, format="espresso-out", index=-1)  # type: ignore[no-untyped-call]
        except Exception as e:
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

        return DFTResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            work_dir=work_dir,
            duration_seconds=duration,
            log_content=content[-1000:],  # Keep tail
            energy=float(energy),
            forces=np.array(forces),
            stress=stress,
            magmoms=magmoms,
        )

    @staticmethod
    def _check_errors(content: str) -> None:
        """
        Check for known error patterns in the output content.
        Raises specific DFTErrors.
        """
        # SCF Convergence
        if "convergence not achieved" in content:
            raise SCFError("SCF convergence not achieved.")

        # Walltime
        if "Maximum CPU time exceeded" in content or "time limit" in content.lower():
            raise WalltimeError("Maximum CPU time exceeded.")

        # Memory
        if "cannot allocate" in content.lower() or "out of memory" in content.lower():
            raise MemoryError("Memory allocation failed.")

        # Generic errors
        if "Error in routine" in content:
            pass

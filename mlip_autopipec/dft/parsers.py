from pathlib import Path

import ase.io
import numpy as np

from mlip_autopipec.core.exceptions import DFTConvergenceError, DFTRuntimeError
from mlip_autopipec.core.models import DFTResult


def parse_pw_output(output_file: Path) -> DFTResult:
    """
    Parses a Quantum Espresso output file.
    Checks for completion and validity of forces.

    Args:
        output_file: Path to the pw.out file.

    Returns:
        DFTResult object containing energy, forces, and stress.

    Raises:
        DFTRuntimeError: If the job failed, crashed, or forces are invalid.
        DFTConvergenceError: If SCF did not converge.
    """
    if not output_file.exists():
        msg = f"Output file {output_file} not found."
        raise DFTRuntimeError(msg)

    # Check for completion
    with output_file.open("r", errors="replace") as f:
        content = f.read()

    if "JOB DONE" not in content:
        # Check if it's convergence error or crash
        if "convergence not achieved" in content:
            msg = "SCF convergence not achieved."
            raise DFTConvergenceError(msg)
        msg = "DFT calculation failed or incomplete (missing 'JOB DONE')."
        raise DFTRuntimeError(msg)

    try:
        # Use ASE to parse
        # ase.io.read returns Atoms or list of Atoms. Default index is -1 (last image).
        atoms = ase.io.read(output_file, format="espresso-out")
        if isinstance(atoms, list):
            atoms = atoms[-1]
    except Exception as e:
        msg = f"Failed to parse DFT output: {e}"
        raise DFTRuntimeError(msg) from e

    # Extract properties
    # ase.io.espresso usually parses energy and forces.
    try:
        energy = atoms.get_potential_energy()
    except Exception as e:
        msg = "Failed to extract energy."
        raise DFTRuntimeError(msg) from e

    try:
        forces = atoms.get_forces()
    except Exception as e:
        msg = "Failed to extract forces."
        raise DFTRuntimeError(msg) from e

    try:
        stress = atoms.get_stress()
    except Exception:
        # Stress might not be present if tstress=False or not parsed
        stress = None

    # Sanity check
    if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
        msg = "Forces contain NaN or Inf values."
        raise DFTRuntimeError(msg)

    return DFTResult(energy=energy, forces=forces, stress=stress)

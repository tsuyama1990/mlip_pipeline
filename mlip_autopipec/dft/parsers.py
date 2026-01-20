"""
Parsers for Quantum Espresso output.
"""
from pathlib import Path

import numpy as np
from ase.io import read

from mlip_autopipec.core.exceptions import DFTConvergenceError, DFTRuntimeError
from mlip_autopipec.core.models import DFTResult


def parse_pw_output(output_path: Path) -> DFTResult:
    """
    Parses the Quantum Espresso output file.

    Args:
        output_path: Path to the output file (e.g., pw.out).

    Returns:
        DFTResult: The parsed results.

    Raises:
        DFTRuntimeError: If the job failed or crashed.
        DFTConvergenceError: If convergence was not reached (checked via logic).
        FileNotFoundError: If file is missing.
    """
    if not output_path.exists():
        msg = f"Output file not found: {output_path}"
        raise FileNotFoundError(msg)

    # Check for "JOB DONE"
    try:
        content = output_path.read_text()
    except Exception as e:
        msg = f"Could not read output file: {e}"
        raise DFTRuntimeError(msg) from e

    if "JOB DONE" not in content:
        # It could be a crash or just didn't finish
        if "convergence not achieved" in content:
             msg = "DFT convergence not achieved."
             raise DFTConvergenceError(msg)
        msg = "DFT job did not complete successfully (JOB DONE not found)."
        raise DFTRuntimeError(msg)

    # Use ASE to parse properties
    try:
        # index=-1 gets the last step
        atoms = read(output_path, format="espresso-out")
    except Exception as e:
        msg = f"ASE failed to parse output: {e}"
        raise DFTRuntimeError(msg) from e

    # Validation of properties
    # DFTResult expects energy, forces, stress
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        # Stress is optional in ASE if not calculated, but we requested tstress=True.
        # ASE might return None or zeros if not found?
        # atoms.get_stress() returns array of 6 elements (Voigt) or 3x3.
        # DFTResult model expects it.
        try:
             stress = atoms.get_stress(voigt=False)
        except Exception: # Not found
             # If stress missing but requested, is it an error?
             # Spec says "result should contain valid 'energy', 'forces', and 'stress'"
             # We can default to zeros or fail. I'll fail to be strict.
             msg = "Stress not found in output."
             raise DFTRuntimeError(msg) from None

    except Exception as e:
         msg = f"Failed to extract properties: {e}"
         raise DFTRuntimeError(msg) from e

    # Check for NaN/Inf
    if np.isnan(energy) or np.isinf(energy):
         msg = "Energy is NaN or Inf."
         raise DFTRuntimeError(msg)
    if np.any(np.isnan(forces)) or np.any(np.isinf(forces)):
         msg = "Forces contain NaN or Inf."
         raise DFTRuntimeError(msg)

    return DFTResult(
        atoms=atoms,
        energy=float(energy),
        forces=forces,
        stress=stress
    )

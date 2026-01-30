import re

import numpy as np

from mlip_autopipec.domain_models.calculation import DFTResult, SCFError, DFTError, WalltimeError

# Conversion factors (Ry to eV, etc.)
# 1 Ry = 13.6056980659 eV
RY_TO_EV = 13.6056980659
# 1 Bohr = 0.529177210903 Angstrom
BOHR_TO_A = 0.529177210903
# Force: Ry/Bohr -> eV/A
FORCE_FACTOR = RY_TO_EV / BOHR_TO_A
# Stress: Ry/Bohr^3 -> eV/A^3
STRESS_FACTOR = RY_TO_EV / (BOHR_TO_A ** 3)


class DFTParser:
    """Parses Quantum Espresso text output."""

    def parse(self, content: str) -> DFTResult:
        """
        Parses the stdout content of a PWSCF run.
        """
        # 1. Error Detection
        if "convergence not achieved" in content:
            raise SCFError("SCF convergence not achieved")

        if "maximum CPU time exceeded" in content:
            raise WalltimeError("Maximum walltime exceeded")

        if "Out of memory" in content or "allocate" in content and "failed" in content:
            # Simple heuristic
            pass # Usually handled by scheduler, but QE might print it.

        # General error catch
        # "Error in routine"
        error_match = re.search(r"Error in routine\s+(\w+)\s+\((.+)\):", content)
        if error_match:
            # Check context lines for specific errors if needed
            msg = f"Error in routine {error_match.group(1)}: {error_match.group(2)}"
            # If followed by specific message
            error_msg_match = re.search(r"Error in routine.*:\s*\n\s+(.+)", content, re.MULTILINE)
            if error_msg_match:
                msg += f" - {error_msg_match.group(1).strip()}"

            if "convergence" in msg:
                raise SCFError(msg)
            raise DFTError(msg)

        # 2. Extract Values

        # Energy
        # !    total energy              =     -15.00000000 Ry
        energy_match = re.search(r"!\s+total energy\s+=\s+([-+]?\d*\.\d+)\s+Ry", content)
        if not energy_match:
            raise DFTError("Energy not found in output")

        energy_ry = float(energy_match.group(1))
        energy_ev = energy_ry * RY_TO_EV

        # Forces
        # Forces acting on atoms (cartesian axes, Ry/au):
        # atom    1 type  1   force =     0.00000000    0.00000000    0.00000000

        forces: list[list[float]] = []
        lines = content.split("\n")
        in_forces = False
        for line in lines:
            if "Forces acting on atoms" in line:
                in_forces = True
                forces = [] # Reset if multiple steps (we want last)
                continue

            if in_forces:
                if "atom" in line and "force =" in line:
                    # atom    1 type  1   force =     0.00000000    0.00000000    0.00000000
                    parts = line.split("force =")[1].split()
                    f = [float(x) for x in parts[:3]]
                    forces.append(f)
                elif "total   stress" in line or "The non-local contrib." in line or not line.strip():
                    if forces:
                        in_forces = False

        if not forces:
            # Maybe relax calculation? Or single point?
            # Static calc should have forces if tprnfor=true
            # If not found, maybe just 0? No, should be error or warning.
            # Assuming structure has atoms.
            pass

        forces_arr = np.array(forces) * FORCE_FACTOR

        # Stress
        #      total   stress  (Ry/bohr**3)                   (kbar)     P=   -0.00
        #   -0.00000000   0.00000000   0.00000000        -0.00      0.00      0.00
        stress: list[list[float]] = []
        for i, line in enumerate(lines):
            if "total   stress" in line and "(Ry/bohr**3)" in line:
                # Next 3 lines
                stress_lines = lines[i+1 : i+4]
                stress_block = []
                try:
                    for s_line in stress_lines:
                        parts = s_line.split()
                        # First 3 are Ry/bohr^3, next 3 are kbar
                        row = [float(x) for x in parts[:3]]
                        stress_block.append(row)
                    # Use a temp variable for array to avoid typing conflict
                    stress = stress_block
                except (ValueError, IndexError):
                    pass
                break

        stress_arr = None
        if len(stress) == 3:
            stress_arr = np.array(stress) * STRESS_FACTOR

        return DFTResult(
            energy=energy_ev,
            forces=forces_arr,
            stress=stress_arr,
            converged=True
        )

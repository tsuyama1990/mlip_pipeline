import re
import numpy as np

from mlip_autopipec.domain_models.calculation import (
    SCFError,
    DFTError,
    WalltimeError,
    MemoryError,
)


class DFTParser:
    RY_TO_EV = 13.6056980659
    BOHR_TO_A = 0.529177210903
    FORCE_UNIT = RY_TO_EV / BOHR_TO_A
    STRESS_UNIT = RY_TO_EV / (BOHR_TO_A**3)

    def parse_output(self, stdout: str, stderr: str = "") -> dict:
        """
        Parses the output of a QE run.
        Returns a dict of results (energy, forces, etc.) or raises an error.
        """
        # 1. Check for specific errors
        if "convergence not achieved" in stdout:
            raise SCFError("convergence not achieved")

        if (
            "maximum CPU time exceeded" in stdout
            or "maximum CPU time exceeded" in stderr
        ):
            raise WalltimeError("Maximum CPU time exceeded")

        if "Out Of Memory" in stderr or "paramemory" in stderr:  # Common patterns
            raise MemoryError("Out of Memory")

        # 2. Extract Energy
        # !    total energy              =     -150.12345678 Ry
        energy_match = re.search(r"!\s+total energy\s+=\s+([-+]?\d*\.\d+)", stdout)
        if not energy_match:
            if "JOB DONE" not in stdout and not stderr:
                # If we don't have errors but job didn't finish cleanly
                raise DFTError("Job incomplete or crashed without specific error.")
            if "JOB DONE" in stdout:
                raise DFTError("Job done but energy not found.")
            raise DFTError("Unknown error (no energy found).")

        energy_ry = float(energy_match.group(1))
        energy_ev = energy_ry * self.RY_TO_EV

        # 3. Extract Forces
        # atom    1 type  1   force =     0.00100000    0.00200000    0.00300000
        forces_matches = re.findall(
            r"force\s+=\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)", stdout
        )
        if not forces_matches:
            # Forces are optional? Spec says DFTResult must have forces.
            # QE only prints forces if tprnfor=.true.
            raise DFTError("Forces not found (check tprnfor=.true.)")

        forces_ryau = np.array(forces_matches, dtype=float)
        forces_evang = forces_ryau * self.FORCE_UNIT

        # 4. Extract Stress
        # stress (Ry/bohr**3)                   (kbar)     P=   -0.12
        #   -0.00010000   0.00000000   0.00000000       -14.70        0.00        0.00
        # ...
        stress_match = re.search(
            r"stress \(Ry/bohr\*\*3\).*?\n\s*"
            r"([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+).*\n\s*"
            r"([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+).*\n\s*"
            r"([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)",
            stdout,
        )

        stress_evang = None
        if stress_match:
            stress_flat = [float(x) for x in stress_match.groups()]
            stress_rybohr = np.array(stress_flat).reshape(3, 3)
            stress_evang = stress_rybohr * self.STRESS_UNIT

        return {"energy": energy_ev, "forces": forces_evang, "stress": stress_evang}

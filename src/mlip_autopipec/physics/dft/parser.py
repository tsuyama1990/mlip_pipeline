import re
from pathlib import Path
from typing import Optional

import numpy as np

from mlip_autopipec.domain_models.calculation import (
    DFTResult,
    DFTError,
    SCFError,
    MemoryError,
    WalltimeError,
)
from mlip_autopipec.domain_models.job import JobStatus


class DFTParser:
    """
    Parses Quantum Espresso output.
    """

    # Constants from ASE units (or similar standard)
    RY_TO_EV = 13.6056980659
    BOHR_TO_A = 0.529177210903

    def parse(self, output: str, job_id: str = "unknown", work_dir: Path = Path(".")) -> DFTResult:
        """
        Parse the standard output string from pw.x.
        """

        # 1. Check for errors
        self._check_errors(output)

        # 2. Extract results
        energy = self._extract_energy(output)
        forces = self._extract_forces(output)
        stress = self._extract_stress(output)

        # If we reached here without error but energy is missing, something is wrong
        if energy is None:
             # Check if job was interrupted without explicit error message
             if "JOB DONE" not in output:
                 # Assume it died unexpectedly (maybe OOM or timeout caught by shell but not printed nicely)
                 # But usually we check errors first.
                 raise DFTError("Job finished but no energy found and no specific error detected.")

        # If forces are missing but energy is present, maybe tprnfor was False?
        # But we enforce tprnfor=True.
        if forces is None:
             raise DFTError("Forces not found in output.")

        return DFTResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            work_dir=work_dir,
            duration_seconds=self._extract_duration(output),
            log_content=output[-1000:], # Keep tail
            energy=energy,
            forces=forces,
            stress=stress
        )

    def _check_errors(self, output: str):
        if "convergence not achieved" in output:
            raise SCFError("SCF convergence not achieved")

        if "maximum CPU time exceeded" in output or "stopping ... found" in output:
             # "stopping ... found" usually means soft exit due to max_seconds
             raise WalltimeError("Maximum CPU time exceeded")

        if "Error in routine" in output:
            # Generic catch, try to identify specific
            if "allocation" in output or "allocate" in output:
                raise MemoryError("Memory allocation failed")
            if "c_bands" in output and "too many bands are not converged" in output:
                 # This is essentially an SCF/Diagonalization issue
                 raise SCFError("Too many bands not converged")

            # Default to generic DFTError with the message
            match = re.search(r"Error in routine\s+(\w+)\s+\(\d+\):\s+(.+)", output)
            msg = match.group(2).strip() if match else "Unknown QE Error"
            raise DFTError(f"QE Error: {msg}")

    def _extract_energy(self, output: str) -> Optional[float]:
        # !    total energy              =     -15.78901234 Ry
        match = re.search(r"!\s+total energy\s+=\s+([-+]?\d*\.\d+)\s+Ry", output)
        if match:
            return float(match.group(1)) * self.RY_TO_EV
        return None

    def _extract_forces(self, output: str) -> Optional[np.ndarray]:
        # Forces acting on atoms (cartesian axes, Ry/au):
        # atom    1 type  1   force =     0.00000000    0.00000000    0.00120000

        lines = output.split('\n')
        forces = []
        parsing = False

        for line in lines:
            if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
                parsing = True
                forces = []
                continue

            if parsing:
                if "atom" in line and "force =" in line:
                    # atom    1 type  1   force =     0.00000000    0.00000000    0.00120000
                    parts = line.split("force =")[1].strip().split()
                    vec = [float(x) for x in parts]
                    forces.append(vec)
                elif "Total force" in line or "The non-local contrib." in line or len(line.strip()) == 0:
                    # End of block (empty line often separates)
                    # But be careful, empty lines can be anywhere.
                    # Usually "Total force" or "stress" follows.
                    if len(forces) > 0 and (len(line.strip()) == 0 or "stress" in line):
                        parsing = False
                        break

        if len(forces) > 0:
            # Convert Ry/au to eV/A
            factor = self.RY_TO_EV / self.BOHR_TO_A
            return np.array(forces) * factor
        return None

    def _extract_stress(self, output: str) -> Optional[np.ndarray]:
        # stress (Ry/bohr**3)                   (kbar)     P=   -0.50
        # -0.00000330   0.00000000   0.00000000        -0.49      0.00      0.00

        lines = output.split('\n')
        stress = []
        parsing = False

        for line in lines:
            if "stress (Ry/bohr**3)" in line:
                parsing = True
                stress = []
                continue

            if parsing:
                # Expect 3 lines of 3 floats (first 3 columns)
                # content: -0.00000330   0.00000000   0.00000000        -0.49      0.00      0.00
                parts = line.split()
                if len(parts) >= 6: # 3 Ry/bohr, 3 kbar
                    vec = [float(x) for x in parts[:3]]
                    stress.append(vec)
                    if len(stress) == 3:
                        break

        if len(stress) == 3:
             # Convert Ry/bohr^3 to eV/A^3
             factor = self.RY_TO_EV / (self.BOHR_TO_A**3)
             return np.array(stress) * factor
        return None

    def _extract_duration(self, output: str) -> float:
        # PWSCF        :     1.20s CPU     1.30s WALL
        # or
        # total cpu time spent up to now is        1.2 secs

        # Try finding final time report
        match = re.search(r"PWSCF\s+:\s+[\d\.s]+\s+CPU\s+([\d\.s]+)\s+WALL", output)
        if match:
             # Parse 1.30s -> 1.30
             s = match.group(1).replace("s", "")
             # It might be "1m20s", handling that is harder.
             # QE format is usually XhYmZs or just Xs
             return self._parse_time(s)

        # Alternative: total cpu time spent up to now is ...
        # But this appears multiple times. We want the last one.
        matches = re.findall(r"total cpu time spent up to now is\s+([\d\.]+)\s+secs", output)
        if matches:
            return float(matches[-1])

        return 0.0

    def _parse_time(self, time_str: str) -> float:
        """Helper to parse QE time string like 1h20m3s or 1.20s"""
        # Simple implementation for now, assuming seconds if 's' is at end
        # Or just float
        try:
             return float(time_str)
        except ValueError:
             # Handle m/h
             total = 0.0
             if 'h' in time_str:
                 parts = time_str.split('h')
                 total += float(parts[0]) * 3600
                 time_str = parts[1]
             if 'm' in time_str:
                 parts = time_str.split('m')
                 total += float(parts[0]) * 60
                 time_str = parts[1]
             if 's' in time_str:
                 time_str = time_str.replace('s', '')
                 if time_str:
                    total += float(time_str)
             return total

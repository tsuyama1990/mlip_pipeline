import re

from ase.units import Bohr, Ry


class DFTConvergenceError(Exception):
    pass


class QEParser:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def parse_energy(self) -> float:
        """Parses total energy from QE output. Returns energy in eV."""
        # !    total energy              =     -156.23456789 Ry
        match = re.search(r"!\s+total energy\s+=\s+([-+]?\d*\.\d+)\s+Ry", self.stdout)
        if match:
            ry_val = float(match.group(1))
            return ry_val * Ry
        msg = "Could not find energy in output"
        raise ValueError(msg)

    def parse_forces(self) -> list[list[float]]:
        """Parses forces from QE output. Returns forces in eV/Angstrom."""
        # Forces acting on atoms (cartesian axes, Ry/au):
        # Ry/au to eV/Angstrom
        conversion = Ry / Bohr

        forces: list[list[float]] = []
        lines = self.stdout.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if "Forces acting on atoms (cartesian axes, Ry/au):" in line:
                start_idx = i + 2  # Skip blank line
                break

        if start_idx == -1:
            return []

        for line in lines[start_idx:]:
            if "atom" not in line or "force =" not in line:
                # Check for end of block
                if line.strip() == "" and len(forces) > 0:
                    break
                if "atom" not in line and len(forces) > 0:
                    break
                continue

            # atom    1 type  1   force =     0.10000000    0.00000000    0.00000000
            try:
                parts = line.split("force =")[1].strip().split()
                f = [float(x) * conversion for x in parts]
                forces.append(f)
            except (IndexError, ValueError):
                continue

        return forces

    def parse_stress(self) -> list[list[float]] | None:
        """Parses stress from QE output. Returns stress in eV/Angstrom^3."""
        # total   stress  (Ry/bohr**3)                   (kB)
        conversion = Ry / (Bohr**3)

        lines = self.stdout.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            if "total   stress  (Ry/bohr**3)" in line:
                start_idx = i + 1
                break

        if start_idx == -1:
            return None

        stress: list[list[float]] = []
        # Next 3 lines
        for j in range(3):
            if start_idx + j >= len(lines):
                break
            line = lines[start_idx + j]
            # -0.00000010   0.00000000   0.00000000        -0.01      0.00      0.00
            # First 3 are Ry/bohr^3
            try:
                parts = line.split()[:3]
                if len(parts) < 3:
                    break
                row = [float(x) * conversion for x in parts]
                stress.append(row)
            except ValueError:
                break

        if len(stress) != 3:
            return None
        return stress

    def check_convergence(self) -> None:
        """Checks for convergence errors in output."""
        stdout_lower = self.stdout.lower()
        if "convergence not achieved" in stdout_lower:
            msg = "SCF convergence not achieved"
            raise DFTConvergenceError(msg)

        # Check for other common errors
        if "Error in routine" in self.stdout:
            # Find which routine
            match = re.search(r"Error in routine\s+(\w+)", self.stdout)
            routine = match.group(1) if match else "unknown"
            msg = f"QE Error in routine {routine}"
            raise DFTConvergenceError(msg)

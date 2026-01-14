import numpy as np

from mlip_autopipec.schemas.dft import DFTOutput


class QEOutputParser:
    """
    Parses the output of a Quantum Espresso calculation.

    This class takes the stdout of a QE calculation and parses it to
    extract the total energy, forces, and stress.
    """

    def __init__(self, qe_output: str) -> None:
        """
        Initializes the QEOutputParser.

        Args:
            qe_output: The stdout of the QE calculation.
        """
        self.qe_output = qe_output

    def parse(self) -> DFTOutput:
        """
        Parses the QE output and returns a DFTOutput object.

        Returns:
            The parsed DFT output.
        """
        total_energy = 0.0
        forces: list[list[float]] = []
        stress: list[list[float]] = []
        parsing_forces = False
        parsing_stress = False
        stress_lines: list[str] = []

        for line in self.qe_output.splitlines():
            if "!    total energy" in line:
                total_energy = float(line.split()[-2])
            elif "Forces acting on atoms" in line:
                parsing_forces = True
            elif parsing_forces and "atom" in line:
                parts = line.split()
                forces.append([float(parts[6]), float(parts[7]), float(parts[8])])
            elif "total stress" in line:
                parsing_forces = False
                parsing_stress = True
                stress_lines = []
            elif parsing_stress and len(stress_lines) < 3:
                stress_lines.append(line)

        if stress_lines:
            stress = np.array([list(map(float, line.split())) for line in stress_lines])
            stress = (stress * 1e-1).tolist()  # Convert from kbar to GPa

        return DFTOutput(total_energy=total_energy, forces=forces, stress=stress)

"""Module for running and managing DFT calculations with Quantum Espresso."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError

from mlip_autopipec.config.system import SystemConfig

# Setup a module-level logger
logger = logging.getLogger(__name__)


class DFTCalculationError(Exception):
    """Custom exception for errors during DFT calculations."""


class QEInputGenerator:
    """A class responsible for generating Quantum Espresso input files.

    This class encapsulates the logic for converting an ASE `Atoms` object and a
    `SystemConfig` into a correctly formatted string that can be used as an
    input file for a `pw.x` calculation.
    """

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the QEInputGenerator.

        Args:
            config: The fully-expanded system configuration object, which
                    contains all necessary DFT parameters.

        """
        self.config = config

    def generate(self, atoms: Atoms) -> str:
        """Generate the content of a Quantum Espresso input file.

        This method constructs the input file string by combining various
        sections (`&CONTROL`, `&SYSTEM`, etc.) and cards (`ATOMIC_SPECIES`,
        `ATOMIC_POSITIONS`, etc.) based on the configuration and the provided
        atomic structure.

        Args:
            atoms: The ASE `Atoms` object representing the atomic structure.

        Returns:
            The formatted input file content as a string.

        """
        dft = self.config.dft
        dft.system.nat = len(atoms)
        dft.system.ntyp = len(dft.pseudopotentials.root)

        # Build the input file string section by section
        control_part = self._format_namelist("CONTROL", dft.control.model_dump())
        system_part = self._format_namelist("SYSTEM", dft.system.model_dump())
        electrons_part = self._format_namelist("ELECTRONS", dft.electrons.model_dump())

        species_part = self._format_atomic_species(dft.pseudopotentials.root)
        positions_part = self._format_atomic_positions(atoms)
        kpoints_part = "K_POINTS {automatic}\n  1 1 1 0 0 0\n"
        cell_part = self._format_cell_parameters(atoms)

        return (
            f"{control_part}\n{system_part}\n{electrons_part}\n"
            f"{species_part}\n{positions_part}\n{kpoints_part}\n{cell_part}"
        )

    @staticmethod
    def _format_namelist(name: str, params: dict[str, Any]) -> str:
        """Format a Python dictionary into a QE namelist string."""
        lines = [f"&{name}"]
        for key, value in params.items():
            if value is None:
                continue
            formatted_value = (
                ".true."
                if isinstance(value, bool) and value
                else ".false."
                if isinstance(value, bool) and not value
                else f"'{value}'"
                if isinstance(value, str)
                else str(value)
            )
            lines.append(f"  {key} = {formatted_value}")
        lines.append("/")
        return "\n".join(lines)

    @staticmethod
    def _format_atomic_species(pseudos: dict[str, str]) -> str:
        """Format the ATOMIC_SPECIES card."""
        lines = ["ATOMIC_SPECIES"]
        # A dummy mass is fine for static calculations
        for symbol, pseudo_file in sorted(pseudos.items()):
            lines.append(f"  {symbol} 1.0 {pseudo_file}")
        return "\n".join(lines)

    @staticmethod
    def _format_atomic_positions(atoms: Atoms) -> str:
        """Format the ATOMIC_POSITIONS card."""
        lines = ["ATOMIC_POSITIONS {angstrom}"]
        for atom in atoms:
            pos = " ".join(map(str, atom.position))
            lines.append(f"  {atom.symbol} {pos}")
        return "\n".join(lines)

    @staticmethod
    def _format_cell_parameters(atoms: Atoms) -> str:
        """Format the CELL_PARAMETERS card."""
        lines = ["CELL_PARAMETERS {angstrom}"]
        for vector in atoms.cell:
            lines.append(f"  {' '.join(map(str, vector))}")
        return "\n".join(lines)


class QEFileManager:
    """Manages filesystem operations for a Quantum Espresso calculation."""

    def __init__(self) -> None:
        """Initialize the QEFileManager."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self._temp_dir.name)

    @property
    def input_path(self) -> Path:
        """Path to the input file."""
        return self.work_dir / "dft.in"

    @property
    def output_path(self) -> Path:
        """Path to the output file."""
        return self.work_dir / "dft.out"

    def write_input(self, content: str) -> None:
        """Write the input file."""
        self.input_path.write_text(content)

    def cleanup(self) -> None:
        """Clean up the temporary directory."""
        self._temp_dir.cleanup()


class QEOutputParser:
    """Parses Quantum Espresso output files."""

    def parse(self, output_path: Path) -> dict[str, Any]:
        """Parse the output file from Quantum Espresso to extract results.

        Args:
            output_path: Path to the QE output file.

        Returns:
            A dictionary containing the parsed energy, forces, and stress.

        """
        from ase.io.espresso import read_espresso_out

        with open(output_path) as f:
            # ASE's parser is robust and well-tested
            parsed_atoms_list = list(read_espresso_out(f, index=slice(None)))

        if not parsed_atoms_list:
            raise DFTCalculationError(
                "Failed to parse any configuration from QE output."
            )

        final_atoms = parsed_atoms_list[-1]
        try:
            stress = final_atoms.get_stress(voigt=False)
        except PropertyNotImplementedError:
            logger.warning("Stress tensor not found in QE output. Setting to zeros.")
            stress = np.zeros((3, 3))

        return {
            "energy": final_atoms.get_potential_energy(),
            "forces": final_atoms.get_forces(),
            "stress": stress,
        }


class QEProcessRunner:
    """A robust runner for executing Quantum Espresso (pw.x) calculations."""

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the QEProcessRunner.

        Args:
            config: The fully-expanded system configuration object.

        """
        self.config = config

    def execute(self, input_path: Path, output_path: Path) -> None:
        """Run the pw.x executable as a subprocess.

        Args:
            input_path: Path to the QE input file.
            output_path: Path to write the QE output.

        Raises:
            DFTCalculationError: If pw.x returns a non-zero exit code.

        """
        command = [self.config.dft.command, "-in", str(input_path)]
        logger.info("Executing DFT command: %s", " ".join(command))
        # The use of subprocess.run is secure because `shell=False` is the
        # default and the command is passed as a list, preventing shell
        # injection. Additionally, `tempfile.TemporaryDirectory` creates a
        # secure, private directory, and `pathlib` joins prevent path
        # traversal attacks (`../`), ensuring files are written within the
        # temporary directory.
        try:
            with open(output_path, "w") as f:
                subprocess.run(
                    command,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
        except FileNotFoundError as e:
            error_message = (
                f"DFT command '{self.config.dft.command}' not found. "
                "Ensure Quantum Espresso is installed and in the system's PATH."
            )
            logger.error(error_message)
            raise DFTCalculationError(error_message) from e
        except subprocess.CalledProcessError as e:
            error_message = (
                f"DFT calculation failed with exit code {e.returncode}.\n"
                f"  Input file: {input_path}\n"
                f"  Output file: {output_path}\n"
                f"  Stderr: {e.stderr}"
            )
            logger.error(error_message)
            raise DFTCalculationError(error_message) from e
        logger.info("DFT calculation finished successfully. Output at %s", output_path)


class DFTFactory:
    """A factory for running DFT calculations."""

    def __init__(self, config: SystemConfig) -> None:
        """Initialize the DFTFactory.

        Args:
            config: The fully-expanded system configuration object.

        """
        self.config = config
        self.input_generator = QEInputGenerator(config)
        self.process_runner = QEProcessRunner(config)
        self.output_parser = QEOutputParser()

    def run(self, atoms: Atoms) -> Atoms:
        """Run a DFT calculation.

        Args:
            atoms: The ASE `Atoms` object representing the structure.

        Returns:
            The input `Atoms` object with calculation results attached.

        """
        file_manager = QEFileManager()
        input_content = self.input_generator.generate(atoms)
        file_manager.write_input(input_content)

        self.process_runner.execute(file_manager.input_path, file_manager.output_path)

        results = self.output_parser.parse(file_manager.output_path)
        atoms.calc.results = results

        file_manager.cleanup()
        return atoms

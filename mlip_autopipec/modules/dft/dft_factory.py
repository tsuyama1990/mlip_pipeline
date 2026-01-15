# ruff: noqa: D101, D102, D103, D107
"""Module for running Quantum Espresso simulations."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Any
import tempfile
import shutil
import re
import logging
from ase import Atoms
from mlip_autopipec.config_schemas import SystemConfig, DFTInput
class DFTCalculationError(Exception):
    """Custom exception for errors during DFT calculations."""
logger = logging.getLogger(__name__)
class QEProcessRunner:
    """A robust wrapper for executing Quantum Espresso calculations."""
    def __init__(self, config: SystemConfig, input_generator: QEInputGenerator, output_parser: QEOutputParser) -> None:
        """Initialize the QEProcessRunner."""
        self.config = config
        self.input_generator = input_generator
        self.output_parser = output_parser
    def run(self, atoms: Atoms) -> Atoms:
        """Execute and manage a Quantum Espresso calculation with auto-recovery."""
        dft_input = self.config.dft.input.model_copy(deep=True)
        for attempt in range(self.config.dft.max_retries + 1):
            try:
                logger.info(f"Starting DFT calculation (Attempt {attempt + 1}/{self.config.dft.max_retries + 1})...")
                return self._perform_calculation(atoms, dft_input)
            except (DFTCalculationError, subprocess.CalledProcessError) as e:
                logger.warning(f"DFT calculation attempt {attempt + 1} failed: {e}")
                if attempt < self.config.dft.max_retries:
                    if attempt < len(self.config.dft.retry_strategy):
                        retry_params = self.config.dft.retry_strategy[attempt]
                        dft_input = self._apply_retry_strategy(dft_input, retry_params)
                    else:
                        logger.warning("No more specific retry strategies. Retrying with same parameters.")
                else:
                    raise DFTCalculationError("All DFT calculation attempts failed.") from e
        raise DFTCalculationError("All DFT calculation attempts failed.")
    def _perform_calculation(self, atoms: Atoms, dft_input: DFTInput) -> Atoms:
        """Perform a single Quantum Espresso calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "qe_input.in"
            output_file = temp_path / "qe_output.out"
            self.input_generator.generate_input(input_file, atoms, dft_input)
            self._execute_pw_x(input_file, output_file, temp_path)
            return self.output_parser.parse_output(output_file, atoms)
    def _execute_pw_x(
        self, input_file: Path, output_file: Path, work_dir: Path
    ) -> None:
        """Execute the pw.x command securely."""
        executable = self.config.dft.executable.command
        if not shutil.which(executable):
            raise FileNotFoundError(f"Executable '{executable}' not found.")
        command = [executable, "-in", str(input_file)]
        with open(output_file, "w") as f_out:
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=work_dir,
                )
            except subprocess.CalledProcessError as e:
                raise DFTCalculationError(f"QE execution failed: {e.stderr}") from e
    def _apply_retry_strategy(
        self, dft_input: DFTInput, strategy: dict[str, Any]
    ) -> DFTInput:
        """Apply the retry strategy to the DFT input parameters."""
        logger.info(f"Applying retry strategy: {strategy}")
        new_input = dft_input.model_copy(deep=True)
        for section, params in strategy.get("params", {}).items():
            if hasattr(new_input, section):
                section_obj = getattr(new_input, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        return new_input
class QEInputGenerator:
    """Generates input files for Quantum Espresso."""
    def generate_input(
        self, file_path: Path, atoms: Atoms, dft_input: DFTInput
    ) -> None:
        """Generate a complete Quantum Espresso input file."""
        content = self._generate_control_namelist(dft_input.control)
        content += self._generate_system_namelist(atoms, dft_input.system)
        content += self._generate_electrons_namelist(dft_input.electrons)
        content += self._generate_atomic_species_card(atoms, dft_input.pseudopotentials)
        content += self._generate_atomic_positions_card(atoms)
        content += self._generate_k_points_card()
        content += self._generate_cell_parameters_card(atoms)
        file_path.write_text(content)
    def _generate_control_namelist(self, control_params: dict[str, Any]) -> str:
        return self._format_namelist("CONTROL", control_params)
    def _generate_system_namelist(
        self, atoms: Atoms, system_params: dict[str, Any]
    ) -> str:
        system_params.nat = len(atoms)
        system_params.ntyp = len(set(atoms.get_chemical_symbols()))
        return self._format_namelist("SYSTEM", system_params)
    def _generate_electrons_namelist(self, electrons_params: dict[str, Any]) -> str:
        return self._format_namelist("ELECTRONS", electrons_params)
    def _format_namelist(self, name: str, params: dict[str, Any]) -> str:
        lines = [f"&{name}"]
        for key, value in params.model_dump().items():
            if value is not None:
                lines.append(f"  {key} = {self._format_value(value)}")
        lines.append("/")
        return "\n".join(lines) + "\n"
    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            return f"'{value}'"
        if isinstance(value, bool):
            return f".{str(value).lower()}."
        return str(value)
    def _generate_atomic_species_card(
        self, atoms: Atoms, pseudopotentials: dict[str, str]
    ) -> str:
        lines = ["ATOMIC_SPECIES"]
        symbols = sorted(list(set(atoms.get_chemical_symbols())))
        masses = {symbol: atoms.get_masses()[list(atoms.get_chemical_symbols()).index(symbol)] for symbol in symbols}
        for symbol in symbols:
            lines.append(
                f"  {symbol} {masses[symbol]:.4f} {pseudopotentials[symbol]}"
            )
        return "\n".join(lines) + "\n"
    def _generate_atomic_positions_card(self, atoms: Atoms) -> str:
        lines = ["ATOMIC_POSITIONS {angstrom}"]
        for atom in atoms:
            lines.append(
                f"  {atom.symbol} {atom.position[0]:.8f} {atom.position[1]:.8f} {atom.position[2]:.8f}"
            )
        return "\n".join(lines) + "\n"
    def _generate_k_points_card(self) -> str:
        return "K_POINTS {gamma}\n"
    def _generate_cell_parameters_card(self, atoms: Atoms) -> str:
        lines = ["CELL_PARAMETERS {angstrom}"]
        for vector in atoms.get_cell():
            lines.append(f"  {vector[0]:.8f} {vector[1]:.8f} {vector[2]:.8f}")
        return "\n".join(lines) + "\n"
class QEOutputParser:
    """Parses output files from Quantum Espresso to extract calculation results."""
    def parse_output(self, file_path: Path, atoms: Atoms) -> Atoms:
        """Parse the QE output file to extract energy, forces, and stress."""
        content = file_path.read_text()
        atoms.info["energy"] = self._parse_energy(content)
        atoms.arrays["forces"] = self._parse_forces(content)
        atoms.info["stress"] = self._parse_stress(content)
        return atoms
    def _parse_energy(self, content: str) -> float:
        match = re.search(r"!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", content)
        if not match:
            raise DFTCalculationError("Could not parse total energy from QE output.")
        return float(match.group(1)) * 13.6057  # Ry to eV
    def _parse_forces(self, content: str) -> list[list[float]]:
        match = re.search(
            r"Forces acting on atoms \(cartesian axes, Ry\/au\):\s*\n(.*?)\n\n",
            content,
            re.DOTALL,
        )
        if not match:
            raise DFTCalculationError("Could not parse forces from QE output.")
        lines = match.group(1).strip().split("\n")
        forces = []
        for line in lines:
            parts = line.split()
            forces.append([float(p) * 25.711 for p in parts[-3:]])  # Ry/au to eV/A
        return forces
    def _parse_stress(self, content: str) -> list[float]:
        match = re.search(
            r"total\s+stress\s+\(Ry\/bohr\*\*3\)\s+\(kbar\)\s+P=\s*(-?\d+\.\d+)\s*\n"
            r"((?:\s+-?\d+\.\d+)+)",
            content,
        )
        if not match:
            raise DFTCalculationError("Could not parse stress from QE output.")
        stress_matrix_lines = match.group(2).strip().split("\n")
        stress_matrix = [list(map(float, line.split())) for line in stress_matrix_lines]
        voigt_stress = [
            stress_matrix[0][0],
            stress_matrix[1][1],
            stress_matrix[2][2],
            stress_matrix[1][2],
            stress_matrix[0][2],
            stress_matrix[0][1],
        ]
        return [s * -14705.5 for s in voigt_stress]  # kbar to eV/A^3 and correct sign

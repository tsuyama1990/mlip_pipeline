"""Module for generating Quantum Espresso input files."""

from typing import Any

from ase import Atoms

from mlip_autopipec.config_schemas import DFTConfig


class QEInputGenerator:
    """A class responsible for generating Quantum Espresso input files.

    This class encapsulates the logic for converting an ASE `Atoms` object and a
    `SystemConfig` into a correctly formatted string that can be used as an
    input file for a `pw.x` calculation.
    """

    def generate(self, atoms: Atoms, config: DFTConfig) -> str:
        """Generate the content of a Quantum Espresso input file.

        This method constructs the input file string by combining various
        sections (`&CONTROL`, `&SYSTEM`, etc.) and cards (`ATOMIC_SPECIES`,
        `ATOMIC_POSITIONS`, etc.) based on the configuration and the provided
        atomic structure.

        Args:
            atoms: The ASE `Atoms` object representing the atomic structure.
            config: The DFT-specific configuration object.

        Returns:
            The formatted input file content as a string.

        """
        dft_input = config.input
        dft_input.system.nat = len(atoms)
        dft_input.system.ntyp = len(dft_input.pseudopotentials)

        # Build the input file string section by section
        control_part = self._format_namelist("CONTROL", dft_input.control.model_dump())
        system_part = self._format_namelist("SYSTEM", dft_input.system.model_dump())
        electrons_part = self._format_namelist(
            "ELECTRONS", dft_input.electrons.model_dump()
        )

        species_part = self._format_atomic_species(dft_input.pseudopotentials)
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
